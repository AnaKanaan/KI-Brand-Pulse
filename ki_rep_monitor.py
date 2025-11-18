# ki_rep_monitor.py
import os, json, time, math, re, uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Set

import pandas as pd
import requests
import tldextract
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Modelle / Endpunkte
# =========================================================
OPENAI_BASE = os.getenv("OPENAI_API_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

# Pass‑A (Antwort) Modell
#
# ChatGPT free now uses the GPT‑5.1 Instant model under the hood.  According to
# the latest OpenAI release notes, GPT‑5.1 Instant is exposed in the API as
# ``gpt-5.1-chat-latest`` while the deeper reasoning variant (GPT‑5.1
# Thinking) is exposed as ``gpt-5.1``【677344943524217†L566-L569】.  We default to
# the Instant variant here to approximate the free ChatGPT experience.  This
# value can be overridden via the ``MODEL_CHAT`` environment variable.
MODEL_CHAT  = os.getenv("MODEL_CHAT", "gpt-5.1-chat-latest")

# Pass‑B (Codierung) Modell
#
# For normalisation we use GPT‑5.1 (Thinking) by default.  This model offers
# deeper reasoning and supports JSON mode.  It can be overridden via
# ``MODEL_PASS_B``.
MODEL_PASSB = os.getenv("MODEL_PASS_B", "gpt-5.1")

# Gemini
# The free Gemini chat experience uses the 2.5 Flash model.  According to
# publicly available comparisons of Gemini and ChatGPT, the free plan gives
# "general access" to Gemini 2.5 Flash while only limited access to Gemini 2.5 Pro is
# available【416161300090141†L360-L365】.  We set this as the default model for
# generative calls below.  This can be overridden via the environment
# variable GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# =========================================================
# Utils
# =========================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def require_env_runtime(var: str) -> str:
    v = os.getenv(var)
    if not v:
        raise RuntimeError(f"{var} not set (use UI).")
    return v

def normalize_domain(d: str) -> str:
    if not d:
        return ""
    try:
        from urllib.parse import urlparse
        h = urlparse(d).hostname or d
    except Exception:
        h = d
    h = h.lower().lstrip("www.")
    ext = tldextract.extract(h)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else h

def freshness_bucket(age_days: Optional[int]) -> str:
    if age_days is None: return "unknown"
    if age_days <= 0:    return "today"
    if age_days <= 7:    return "≤7d"
    if age_days <= 30:   return "≤30d"
    if age_days <= 90:   return "≤90d"
    if age_days <= 365:  return "≤365d"
    return ">365d"

def freshness_index(evidence: List[Dict[str, Any]]) -> float:
    vals = [math.exp(-float(e.get("age_days", 0)) / 90.0)
            for e in evidence if isinstance(e.get("age_days"), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0

def known_freshness_pct(evidence: List[Dict[str, Any]]) -> float:
    return (sum(1 for e in evidence if e.get("age_days") is not None) / len(evidence)) if evidence else 0.0

# =========================================================
# HTTP / LLM
# =========================================================

def openai_responses(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to OpenAI Responses API with lightweight retries. Retries on 429/5xx."""
    key = require_env_runtime("OPENAI_API_KEY")
    url = f"{OPENAI_BASE}/responses"
    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120
            )
            if r.status_code < 400:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(0.8 * attempt); continue
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")
        except Exception as ex:
            last_err = ex
            if attempt < 3:
                time.sleep(0.8 * attempt); continue
            raise RuntimeError(f"OpenAI request failed after {attempt} attempts: {last_err}")

def gemini_generate_text(prompt: str, system: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Generate text using the Gemini API.  The Gemini API key is required and
    obtained from the environment variable ``GEMINI_API_KEY``.  If no key is
    set this function raises a RuntimeError.  The default model is
    ``gemini-2.5-flash`` because the free Gemini chat uses that model【416161300090141†L360-L365】.

    Args:
        prompt: The user prompt to send to the model.
        system: Optional system instruction to prepend to the prompt.
        model: Optional explicit model name; overrides the default.

    Returns:
        The textual reply from the model.

    Raises:
        RuntimeError: If the GEMINI_API_KEY environment variable is not set or the
            HTTP request fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (use UI)")
    mdl = model or DEFAULT_GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={api_key}"
    parts = []
    # include a system message if provided
    if system:
        parts.append({"text": f"[SYSTEM]\n{system}\n"})
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2}
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text}")
    d = r.json()
    try:
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as ex:
        raise RuntimeError(f"Gemini response parse error: {ex}")


def extract_text_from_responses(data: Dict[str, Any]) -> str:
    """
    Extract the main answer text from a Responses API payload.

    The Responses API may return multiple blocks in ``output`` such as
    ``reasoning``, ``web_search_call`` and finally a ``message`` block which
    contains the actual answer as ``output_text``.  This helper searches for
    the first ``message`` block and returns its text.  If nothing suitable is
    found we fall back to the old heuristic and, as a last resort, return the
    JSON-dumped payload to aid debugging.
    """
    if isinstance(data, dict) and data.get("output_text"):
        # Some helper wrappers or future formats may surface output_text
        # directly at the top level.
        return data["output_text"]

    out = data.get("output") if isinstance(data, dict) else None

    # Preferred path: walk the output blocks and locate the first message
    # block with an ``output_text`` part.
    if isinstance(out, list):
        for block in out:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "message":
                continue
            for part in (block.get("content") or []):
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and "text" in part
                ):
                    return part["text"]

    # Fallback to the previous array-based assumption for older or unusual
    # payload shapes.
    try:
        return out[0]["content"][0]["text"]
    except Exception:
        # Last-resort fallback: return the full JSON payload as a string so
        # that it can be inspected in the debug logs / RawAnswers sheet.
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)

def extract_text_and_citations(data: Dict[str, Any]):
    text = extract_text_from_responses(data) or ""
    citations = []
    try:
        for block in (data.get("output") or []):
            if block.get("type") != "message":
                continue
            for part in (block.get("content") or []):
                if part.get("type") == "output_text":
                    for ann in (part.get("annotations") or []):
                        if ann.get("type") == "url_citation":
                            start = int(ann.get("start_index") or 0)
                            end   = int(ann.get("end_index") or 0)
                            snippet = ""
                            try:
                                snippet = text[start:end].strip()
                            except Exception:
                                pass
                            url = ann.get("url") or ""
                            citations.append({
                                "url": url,
                                "title": ann.get("title") or "",
                                "domain": normalize_domain(url),
                                "start": start, "end": end,
                                "snippet": snippet
                            })
    except Exception:
        pass
    dedup = {}
    for c in citations:
        u = c["url"]
        if u not in dedup or len((c.get("snippet") or "")) > len((dedup[u].get("snippet") or "")):
            dedup[u] = c
    return text, list(dedup.values())

# =========================================================
# Citation extraction for plain text (no tool use)
# =========================================================
def extract_domains_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Scan plain text for domain references.  Chat models may include citations
    such as (example.com) or mention full URLs.  This helper extracts domain
    names from any parentheses or word-like patterns containing a dot.  It
    returns a list of evidence dictionaries with the domain and a snippet
    extracted around the domain occurrence.  Duplicate domains are de‑duped.

    Args:
        text: The plain answer text returned from a language model.

    Returns:
        A list of evidence dictionaries with keys 'domain', 'snippet' and
        optionally 'url' (empty because the model doesn't include full URLs).
    """
    evid = []
    if not text:
        return evid
    # find patterns like (example.com) and raw domain names separated by spaces
    pattern = re.compile(r"\(([^\)\s]+\.[a-zA-Z]{2,})\)|\b([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
    seen = set()
    for m in pattern.finditer(text):
        domain = None
        if m.group(1):
            domain = m.group(1)
        elif m.group(2):
            domain = m.group(2)
        if not domain:
            continue
        dom_norm = normalize_domain(domain)
        if not dom_norm or dom_norm in seen:
            continue
        seen.add(dom_norm)
        # capture snippet around the domain (±60 characters)
        start, end = m.span()
        snippet_start = max(0, start - 60)
        snippet_end   = min(len(text), end + 60)
        snippet = text[snippet_start:snippet_end].strip()
        evid.append({
            "title": "",
            "url": "",
            "domain": dom_norm,
            "snippet": snippet,
            "discovered_at": now_iso()
        })
    return evid

def extract_publication_date(url: str) -> Optional[str]:
    """
    Attempt to extract the publication date from a web page.  Many news and
    article pages embed a publication date in meta tags (e.g.
    `article:published_time`, `og:published_time`, or `datePublished`).  If no
    suitable meta tag is found, a generic YYYY‑MM‑DD pattern is searched in
    the HTML.  The returned value is an ISO 8601 timestamp string (UTC) or
    None if no date could be extracted.

    Args:
        url: URL of the web page to fetch.

    Returns:
        ISO 8601 formatted date string (e.g. '2025-05-17T00:00:00') or None.
    """
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code >= 400:
            return None
        html = resp.text
        # search common meta tags
        meta_pattern = re.compile(r'(?:datePublished|article:published_time|og:published_time|article:modified_time)"?\s*content="([^"]+)"', re.IGNORECASE)
        for mt in meta_pattern.finditer(html):
            dt_str = mt.group(1)
            # normalise various separators
            dt_str = dt_str.replace('/', '-').replace('.', '-')
            dt_str = dt_str.strip()
            # if the string contains time, attempt iso parse
            try:
                # replace Z with +00:00 for fromisoformat
                if dt_str.endswith('Z'):
                    dt_str = dt_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(dt_str)
                # unify to UTC iso
                return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
            except Exception:
                pass
        # fallback: search for a date pattern (YYYY-MM-DD)
        date_pattern = re.compile(r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})')
        m = date_pattern.search(html)
        if m:
            ds = m.group(1).replace('/', '-').replace('.', '-')
            parts = ds.split('-')
            # zero pad month/day if necessary
            y, mth, d = parts[0], parts[1].zfill(2), parts[2].zfill(2)
            return f"{y}-{mth}-{d}T00:00:00"
    except Exception:
        return None
    return None

def response_status(data: Dict[str, Any]) -> str:
    try:
        return (data.get("status") or "unknown").lower()
    except Exception:
        return "unknown"

# =========================================================
# Suche & Prompting
# =========================================================
def cse_list(q: str, lang: str, market: str, num: int = 5) -> List[Dict[str, Any]]:
    key = require_env_runtime("GOOGLE_API_KEY")
    cx  = require_env_runtime("GOOGLE_CSE_ID")
    r = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": key, "cx": cx, "q": q, "num": num, "hl": lang, "gl": market},
        timeout=60
    )
    if r.status_code >= 400:
        raise RuntimeError(f"CSE HTTP {r.status_code}: {r.text}")
    items = (r.json().get("items") or [])
    out = []
    for it in items:
        out.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "snippet": it.get("snippet"),
            "displayLink": it.get("displayLink")
        })
    return out

def overview_substitute_with_openai(query: str, cse_items: List[Dict[str, Any]], lang: str,
                                    temperature: float = 0.2, max_tokens: int = 900) -> str:
    sys = f"Antworte in {lang}. Nutze ausschließlich die bereitgestellten Treffer. Zitiere Domains in Klammern, z. B. (nzz.ch)."
    lines = ["Frage:", query, "", "Treffer:"]
    for i, it in enumerate(cse_items, 1):
        domain = it.get("displayLink") or normalize_domain(it.get("link", ""))
        lines.append(f"{i}) {it.get('title','')} — {it.get('link','')} — {domain} — {it.get('snippet','')}")
    user = "\n".join(lines)
    # Note: omit ``temperature`` for models that do not support it (e.g. web search).
    data = openai_responses({
        "model": MODEL_CHAT,
        "input": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        "max_output_tokens": int(max_tokens)
    })
    return extract_text_from_responses(data)

def overview_substitute(query: str, cse_items: List[Dict[str, Any]], lang: str,
                        temperature: float = 0.2, max_tokens: int = 900) -> str:
    combined = []
    for i, it in enumerate(cse_items, 1):
        domain = it.get("displayLink") or normalize_domain(it.get("link", ""))
        combined.append(f"{i}) {it.get('title','')} — {it.get('link','')} — {domain} — {it.get('snippet','')}")
    sys = f"Antworte in {lang}. Nutze ausschließlich die bereitgestellten Treffer. Zitiere Domains in Klammern."
    body = f"{sys}\n\nFrage:\n{query}\n\nTreffer:\n" + "\n".join(combined)
    if os.getenv("GEMINI_API_KEY"):
        txt = gemini_generate_text(prompt=body)
        if txt:
            return txt
    return overview_substitute_with_openai(query, cse_items, lang, temperature, max_tokens)

def render_cse_items_for_coder(items: List[Dict[str, Any]]) -> str:
    lines = []
    for i, e in enumerate(items, 1):
        domain = e.get("domain") or e.get("displayLink") or ""
        url    = e.get("url", e.get("link", ""))
        title  = e.get("title", "")
        snip   = (e.get("snippet") or "").strip()
        if snip:
            lines.append(f"{i}) {title} — {url} — {domain} — {snip}")
        else:
            lines.append(f"{i}) {title} — {url} — {domain}")
    return "\n".join(lines) if lines else "(keine)"

# =========================================================
# Pass B (JSON-Normalisierung)
# =========================================================
def load_coder_prompts(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pass_b_normalize(system_prompt: str, user_prompt: str, model: Optional[str] = None, max_tokens: int = 5000) -> Dict[str, Any]:
    """Run Pass‑B normalisation in JSON mode and attach the raw API response.

    The model is invoked via the Responses API with ``text.format.type =
    'json_object'`` so that it returns a single JSON object.  In addition to
    parsing the JSON payload, this helper stores the complete raw API
    response under ``run_meta.raw_pass_b`` so that it is available in the
    Normalized sheet for later inspection.
    """
    m = model or MODEL_PASSB
    payload = {
        "model": m,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_output_tokens": int(max_tokens),
        # The Responses API expects a nested ``format`` object for structured
        # outputs.  Setting ``type`` to ``json_object`` forces the model to
        # output a valid JSON object.
        "reasoning": {"effort": "medium"},
        "text": {"format": {"type": "json_object"}},
    }
    data = openai_responses(payload)
    txt = extract_text_from_responses(data).strip()
    try:
        obj = json.loads(txt)
    except Exception:
        # tolerate Markdown fenced JSON blocks
        txt2 = re.sub(r"^```json|^```|```$", "", txt, flags=re.M).strip()
        obj = json.loads(txt2)
    # Attach the raw Pass‑B response for full transparency.
    try:
        run_meta = obj.setdefault("run_meta", {})
        run_meta["raw_pass_b"] = data
    except Exception:
        # Never fail the pipeline because of debug metadata issues.
        pass
    return obj

# =========================================================
# Evidence Enrichment
# =========================================================
def enrich_evidence(evidence: List[Dict[str, Any]], domain_seed_csv: str) -> List[Dict[str, Any]]:
    """Enrich evidence items with domain type, dates and freshness.

    Pass‑B is expected to already classify each source into a domain type
    (e.g. ``CORPORATE``, ``NEWS_MEDIA``).  If a model‑assigned
    ``domain_type``/``domain_type_code`` is present it is used as‑is.
    Otherwise a fallback classification based on ``domain_type_seed.csv`` is
    applied.  Publication dates are taken from the evidence object when
    present or, as a fallback, extracted from the URL.
    """
    seeds = pd.read_csv(domain_seed_csv).fillna("")
    direct: Dict[str, str] = {}
    for _, row in seeds.iterrows():
        dt_seed = str(row["domain_type"]).strip()
        for d in str(row.get("example_domains", "")).split(";"):
            d = d.strip()
            if d:
                direct[normalize_domain(d)] = dt_seed

    rules = []
    for _, row in seeds.iterrows():
        dt_seed = str(row["domain_type"]).strip()
        tlds = [t.strip() for t in str(row.get("tld_hints", "")).split(";") if t.strip()]
        kws  = [k.strip().lower() for k in str(row.get("keyword_hints", "")).split(";") if k.strip()]
        if dt_seed:
            rules.append((dt_seed, tlds, kws))

    now = datetime.utcnow()
    out: List[Dict[str, Any]] = []
    for e in evidence:
        domain = normalize_domain(e.get("domain") or e.get("url") or "")
        # 1) Prefer explicit domain type from the model (Pass‑B)
        dt = (str(e.get("domain_type")) or str(e.get("domain_type_code")) or "").strip()
        # 2) Fallback to seed mappings when the model did not decide
        if not dt and domain:
            dt = direct.get(domain, "")
        if not dt and domain:
            text_blob = (e.get("title", "") + " " + e.get("snippet", "")).lower()
            for dtc, tlds, kws in rules:
                if any(domain.endswith(t) for t in tlds if t):
                    dt = dtc
                    break
                if any(k in text_blob for k in kws if k):
                    dt = dtc
                    break
        if not dt:
            dt = "OTHER_UNKNOWN"
        # Human readable label (optional)
        dt_label = (str(e.get("domain_type_label")) or str(e.get("domain_type_readable")) or dt).strip() or dt

        pub = e.get("published_at")
        discovered = e.get("discovered_at") or now_iso()
        # If no publication date is provided attempt to fetch it from the URL
        if not pub:
            url = e.get("url") or e.get("link") or ""
            if url:
                try:
                    pub = extract_publication_date(url)
                except Exception:
                    pub = None
        try:
            pub_dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00")) if pub else None
        except Exception:
            pub_dt = None
        age_days = (now - pub_dt).days if pub_dt else None

        out.append({
            "title": e.get("title"),
            "url": e.get("url") or e.get("link"),
            "domain": domain,
            "domain_type": dt,
            "domain_type_label": dt_label,
            "country": e.get("country") or "unknown",
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "discovered_at": discovered,
            "age_days": age_days,
            "freshness_bucket": freshness_bucket(age_days),
            "snippet": e.get("snippet", ""),
        })
    return out

# =========================================================
# Questions Canon
# =========================================================
def _canonize_questions_df(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q.columns = [str(c).strip() for c in q.columns]
    lower = {c.lower(): c for c in q.columns}
    cmap = {}
    if "id" in lower:     cmap[lower["id"]] = "question_id"
    if "query" in lower:  cmap[lower["query"]] = "question_text"
    for want in ["language", "category", "intent", "variant"]:
        if want not in q.columns and want in lower:
            cmap[lower[want]] = want
    q = q.rename(columns=cmap)

    required = ["question_id", "question_text", "language", "category", "intent", "variant"]
    miss = [c for c in required if c not in q.columns]
    if miss:
        raise KeyError(f"Missing columns in 'Questions': {miss}. Present: {list(q.columns)}")

    q["question_id"] = pd.to_numeric(q["question_id"], errors="coerce").astype("Int64")
    q["intent"]      = pd.to_numeric(q["intent"], errors="coerce").astype("Int64")
    q["variant"]     = pd.to_numeric(q["variant"], errors="coerce").astype("Int64")
    q = q.dropna(subset=["question_id", "question_text", "language"])
    return q

# =========================================================
# Pass A Helper
# =========================================================
def call_chat_no_search(prompt: str, temperature: float = 0.5, max_tokens: int = 900):
    """
    Call the ChatGPT model without explicit web search.  When using the
    Responses API, omitting the ``web_search`` tool allows the model to decide
    whether to use its internal knowledge or perform lightweight retrieval.  We
    request the GPT‑5.1 Instant model (``MODEL_CHAT``) and return the answer
    text along with raw metadata.

    Args:
        prompt: The plain user prompt.
        temperature: Sampling temperature (0.0–1.0).
        max_tokens: Maximum number of output tokens.

    Returns:
        Tuple of (answer_text, meta) where meta contains status and raw
        response data.
    """
    payload = {
        "model": MODEL_CHAT,
        # pass the prompt directly; the Responses API accepts either a string
        # or a list of messages for ``input``.  A single string is sufficient
        # for one-shot interactions.
        "input": prompt,
        "max_output_tokens": int(max_tokens)
    }
    data = openai_responses(payload)
    text = extract_text_from_responses(data)
    return text, {"status": response_status(data), "citations": [], "raw": data}

def call_chat_search_auto(prompt: str,
                          temperature: float = 0.25,
                          max_tokens: int = 4000,
                          search_context_size: str = "medium",
                          user_location: Optional[Dict[str, str]] = None):
    """
    ChatGPT search auto profile using the Responses API web search tool.

    This function invokes the ``web_search`` tool provided by the Responses API
    to perform real‑time web search and returns an answer with citations.  The
    model used is GPT‑5.1 Instant (``MODEL_CHAT``) which aligns with the
    current ChatGPT free experience【677344943524217†L566-L569】.  Starting
    November 2025, ChatGPT’s default backend is GPT‑5.1 and it exposes the
    search tool via the Responses API.  You do not need to use the old
    `web_search_preview` tool from the Chat Completions API.  The
    ``search_context_size`` parameter controls how many search results the
    tool will consider—"low", "medium" (default) or "high"—as documented in
    the AI Engineer guide【486089801306640†L48-L60】.  The optional
    ``user_location`` can be used to localise search results (see docs)
    【486089801306640†L29-L37】.

    Args:
        prompt: Question string for ChatGPT.
        temperature: Sampling temperature for the language model.
        max_tokens: Output token limit.
        search_context_size: Level of search context (``low``, ``medium``, or
            ``high``).  Higher values yield more comprehensive results but are
            slower and costlier【486089801306640†L48-L60】.
        user_location: Optional dictionary specifying approximate location for
            search results (e.g. ``{"type":"approximate", "country":"DE"}``).

    Returns:
        Tuple of (answer_text, meta) where meta includes the status, a list of
        extracted citations, and the raw response data.
    """
    # Build tool configuration
    tool_def: Dict[str, Any] = {"type": "web_search"}
    if search_context_size:
        tool_def["search_context_size"] = search_context_size
    if user_location:
        tool_def["user_location"] = user_location

    # Note: the Responses API interprets ``max_output_tokens`` relative to the
    # selected model.  GPT‑5.1 supports up to ~4 000 output tokens in most
    # configurations.  We default to 4 000 here to accommodate long answers
    # without repeatedly truncating responses.  Adjust via the function
    # parameter if needed.
    # Build the request payload.  Note that the web search models do not
    # currently accept the ``temperature`` parameter, so it is omitted here to
    # prevent invalid_request errors.
    payload = {
        "model": MODEL_CHAT,
        "input": prompt,
        "tools": [tool_def],
        "max_output_tokens": int(max_tokens)
    }
    data = openai_responses(payload)
    # extract answer text and citations using helper
    text, citations = extract_text_and_citations(data)
    return text, {"status": response_status(data), "citations": citations, "raw": data}

# =========================================================
# Gemini Chat calls
# =========================================================
def call_gemini_no_search(prompt: str, temperature: float = 0.2, max_tokens: int = 4000):
    """
    Request a response from the Gemini API without web search.  The free
    Gemini chat uses the 2.5 Flash model【416161300090141†L360-L365】; this function
    delegates to :func:`gemini_generate_text`.  It returns the answer text
    together with metadata.
    """
    try:
        text = gemini_generate_text(prompt, system=None, model=DEFAULT_GEMINI_MODEL)
        return text, {"status": "completed", "raw": text}
    except Exception as ex:
        return f"[ERROR GEMINI: {ex}]", {"status": "error", "error": str(ex)}

def call_gemini_search_auto(prompt: str, lang: str, market: str = "", topn: int = 5,
                             temperature: float = 0.2, max_tokens: int = 4000):
    """
    Query the Gemini API with the built‑in ``google_search`` tool.

    The Gemini API offers a native `google_search` tool that grounds responses
    using real‑time web data【832087404409424†L220-L260】.  This function calls
    ``generateContent`` on the configured Gemini model with the search tool
    enabled.  It then extracts the answer text and evidence (citations) from
    the `groundingMetadata` structure returned by the API.  Unlike earlier
    versions of this function, no Custom Search API is used.  The parameters
    ``lang`` and ``market`` remain for compatibility but are not used, since
    the Gemini search tool automatically determines language and region.

    Args:
        prompt: The user question to send to Gemini.
        lang: Language code (unused but kept for interface consistency).
        market: Region code (unused; the search tool auto-detects context).
        topn: (Unused) kept for backward compatibility.
        temperature: Sampling temperature for Gemini.
        max_tokens: Maximum number of output tokens.

    Returns:
        Tuple (text, meta) where meta contains a list of evidence dictionaries.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[ERROR GEMINI SEARCH: GEMINI_API_KEY not set]", {"status": "error", "error": "GEMINI_API_KEY not set", "citations": []}
    mdl = DEFAULT_GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "tools": [
            {
                "google_search": {}
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens)
        }
    }
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=120)
        if r.status_code >= 400:
            return f"[ERROR GEMINI SEARCH: HTTP {r.status_code}]", {"status": "error", "error": r.text, "citations": []}
        data = r.json()
        # extract answer text
        try:
            cand = data["candidates"][0]
            text = cand["content"]["parts"][0]["text"]
        except Exception as ex:
            return f"[ERROR GEMINI SEARCH PARSE: {ex}]", {"status": "error", "error": str(ex), "citations": []}
        # extract citations from groundingMetadata
        citations: List[Dict[str, Any]] = []
        grounding = cand.get("groundingMetadata") or {}
        chunks = grounding.get("groundingChunks") or []
        for ch in chunks:
            # The chunk may be a dict with key 'web' or a nested message; handle both
            if not isinstance(ch, dict):
                continue
            web = ch.get("web") or {}
            uri = web.get("uri") or ""
            title = web.get("title") or ""
            if uri:
                citations.append({
                    "title": title,
                    "url": uri,
                    "domain": normalize_domain(uri),
                    "snippet": "",
                    "discovered_at": now_iso()
                })
        return text, {"status": "completed", "citations": citations, "raw": data}
    except Exception as ex:
        return f"[ERROR GEMINI SEARCH: {ex}]", {"status": "error", "error": str(ex), "citations": []}

# =========================================================
# Hauptpipeline
# =========================================================


def load_stakeholder_lib(path: str) -> Dict[str, Any]:
    lib = {"map": {}, "langs": {}}
    try:
        xl = pd.ExcelFile(path)
        if "map" in xl.sheet_names:
            dfm = pd.read_excel(xl, sheet_name="map")
            for _, r in dfm.iterrows():
                lib["map"][str(r.get("ui_label_de")).strip()] = str(r.get("stakeholder_id")).strip()
        for lang in ["de","en","fr","it","rm"]:
            if lang in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=lang)
                d = {}
                for _, r in df.iterrows():
                    sid = str(r.get("stakeholder_id")).strip()
                    d[sid] = {
                        "display": ("" if pd.isna(r.get("display")) else str(r.get("display")).strip()),
                        "prefix_template": ("" if pd.isna(r.get("prefix_template")) else str(r.get("prefix_template")))
                    }
                lib["langs"][lang] = d
    except Exception as ex:
        dbg("stakeholder_lib_error", str(ex))
    return lib

def build_stakeholder_prompt(base_q: str, stake_ui_label: str, lang: str, lib_path: str = None) -> str:
    if not lib_path:
        lib_path = os.path.join(BASE_DIR, 'stakeholder_library.xlsx')
    lib = load_stakeholder_lib(lib_path)
    sid = lib["map"].get(stake_ui_label, "generic")
    loc = lib["langs"].get(lang, {}).get(sid, {"display":"", "prefix_template":""})
    display = loc.get("display","")
    prefix_template = loc.get("prefix_template","")
    Q = base_q.replace("<STAKEHOLDER>", display if display else stake_ui_label)
    if "<STAKEHOLDER>" not in base_q and sid != "generic" and prefix_template:
        Q = f"{prefix_template.format(stakeholder=display)} {Q}"
    return Q
def run_pipeline(
    brand: str, topic: str, market: str,
    languages: List[str], profiles: List[str],
    question_xlsx: str, out_xlsx: str,
    domain_seed_csv: str, coder_prompts_json: str,
    topn: int = 5, num_runs: int = 3, categories: Optional[List[str]] = None,
    question_ids: Optional[List[int]] = None,
    comp1: str = "", comp2: str = "", comp3: str = "",
    temperature_chat_no: float = 0.5,
    temperature_chat_search: float = 0.25,
    max_tokens: int = 900,
    wrapper_mode: str = 'free_emulation',
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional["threading.Event"] = None,
    debug_level: str = "verbose",
    max_questions: int = 0,
    passA_search_tokens: Optional[int] = None,
    stakeholders: Optional[List[str]] = None
) -> Dict[str, Any]:

    def emit(ev: Dict[str, Any]):
        if progress:
            try:
                progress(ev)
            except Exception:
                pass

    def dbg(phase: str, msg: str, meta: Optional[Dict[str, Any]] = None):
        if debug_level != "none":
            emit({"t": time.time(), "phase": phase, "msg": msg, "meta": (meta or {})})

    if cancel_event and cancel_event.is_set():
        dbg("cancelled", "Abgebrochen vor Start")
        return {"out": out_xlsx, "runs": 0, "norm": 0, "evi": 0}

    dbg("start", "Lese Questions")
    # Support question libraries organised by language on separate sheets.  If
    # sheet_name "Questions" exists we load it; otherwise we attempt to load
    # each language specific sheet and concatenate them.  Missing sheets are
    # ignored.
    xl = pd.ExcelFile(question_xlsx)
    if "Questions" in xl.sheet_names:
        q = pd.read_excel(xl, sheet_name="Questions")
    else:
        parts = []
        for lang in languages:
            if lang in xl.sheet_names:
                parts.append(pd.read_excel(xl, sheet_name=lang))
        if not parts:
            raise ValueError("Fragenblatt fehlt: weder 'Questions' noch sprachspezifische Blätter gefunden")
        q = pd.concat(parts, ignore_index=True)
    dbg("questions_loaded", f"{len(q)} Zeilen geladen")
    q = _canonize_questions_df(q)

    orig_len = len(q)
    if languages:
        q = q[q["language"].isin(languages)].copy()
    if categories:
        q = q[q["category"].isin(categories)].copy()
    if question_ids:
        q = q[q["question_id"].isin(question_ids)].copy()
    if max_questions and max_questions > 0:
        q = q.head(int(max_questions)).copy()

    dbg("filters_applied", f"Nach Filtern: {len(q)} / {orig_len} Fragen")
    if q.empty:
        raise ValueError("Keine Fragen nach Filter übrig.")

    prompts      = load_coder_prompts(coder_prompts_json)
    system_coder = prompts["system_coder"]
    user_by_lang = prompts["user_coder"]

    runs_rows, norm_rows, ev_rows, raw_rows = [], [], [], []
    cfg_rows = [
        {"key": "wrapper_mode", "value": wrapper_mode},
        {"key": "profiles", "value": ",".join(profiles)}
    ]

    # Determine stakeholders: if none specified use a single generic stakeholder
    stakeholders = [s for s in (stakeholders or []) if s]
    if not stakeholders:
        stakeholders = ["generic"]
    total_jobs = len(q) * max(1, len(profiles)) * max(1, len(stakeholders)) * max(1, int(num_runs))
    done_jobs  = 0

    def check_cancel() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    # iterate through each question
    for _, row in q.iterrows():
        if check_cancel():
            dbg("cancelled", "Abgebrochen")
            break

        # basic question attributes
        qid     = int(row["question_id"])
        intent  = int(row["intent"])
        variant = int(row["variant"])
        lang    = row["language"]

        # prepare base question by substituting placeholders
        base_q = (
            str(row["question_text"])
            .replace("<BRAND>", brand)
            .replace("<TOPIC>", topic)
            .replace("<MARKET>", market)
            .replace("<COMP1>", comp1)
            .replace("<COMP2>", comp2)
            .replace("<COMP3>", comp3)
        )

        # loop profiles, stakeholders and replicates
        for profile in profiles:
            for stake in stakeholders:
                for r in range(int(num_runs)):
                    if check_cancel():
                        dbg("cancelled", "Abgebrochen")
                        break

                    # run identifier
                    run_id = f"{qid}-{profile}-{lang}-{stake}-r{r+1}-{uuid.uuid4().hex[:8]}"

                    # initialise evidence list and default provider/model
                    evidence_list: List[Dict[str, Any]] = []
                    provider = "openai"
                    model_version = MODEL_CHAT

                    # incorporate stakeholder via library mapping
                    Q = build_stakeholder_prompt(base_q, stake, lang, lib_path=os.path.join(BASE_DIR, "stakeholder_library.xlsx"))

                    dbg("build_prompt", f"Frage {qid} / {profile} / r{r+1} Prompt: {Q}")

                    # ---------- PASS A ----------
                    try:
                        if profile == "CHATGPT_NO_SEARCH":
                            dbg("api_call_1_request", "LLM (no search) Anfrage")
                            raw_text, meta_a = call_chat_no_search(Q, temperature=temperature_chat_no, max_tokens=max_tokens)
                        elif profile == "CHATGPT_SEARCH_AUTO":
                            dbg("api_call_1_request", "LLM (search auto) Anfrage")
                            # When performing auto‑search we use a larger token limit.
                            # If ``passA_search_tokens`` is provided we respect it,
                            # otherwise we reuse the ``max_tokens`` parameter (which
                            # defaults to 900 but can be increased via the UI).  We no
                            # longer force a minimum of 1200 tokens because the
                            # default values in the UI are now set to 4 000 to
                            # accommodate longer responses.
                            mtoks = int(passA_search_tokens or max_tokens)
                            # set approximate user location based on the market for better localisation
                            user_loc = None
                            if market:
                                user_loc = {"type": "approximate", "country": market}
                            raw_text, meta_a = call_gemini_search_auto(
                                Q,
                                temperature=temperature_chat_search,
                                max_tokens=mtoks,
                                search_context_size="medium",
                                user_location=user_loc
                            )
                            for c in (meta_a.get("citations") or []):
                                evidence_list.append({
                                    "title": c.get("title"),
                                    "url": c.get("url"),
                                    "domain": c.get("domain"),
                                    "snippet": c.get("snippet", ""),
                                    "discovered_at": now_iso()
                                })
                        elif profile == "GOOGLE_OVERVIEW":
                            dbg("api_call_1_request", "Google CSE")
                            items = cse_list(Q, lang, market, topn)
                            raw_text = overview_substitute(Q, items, lang)
                            for it in items:
                                evidence_list.append({
                                    "title": it.get("title"),
                                    "url": it.get("link"),
                                    "domain": it.get("displayLink") or normalize_domain(it.get("link", "")),
                                    "snippet": it.get("snippet", ""),
                                    "discovered_at": now_iso()
                                })
                            meta_a = {"status": "completed"}
                            provider = "google"
                            model_version = "overview-substitute"
                        elif profile == "GEMINI_NO_SEARCH":
                        # use Gemini wrapper
                            dbg("api_call_1_request", "Gemini (no search) Anfrage")
                            raw_text, meta_g = call_gemini_no_search(Q, temperature=0.2, max_tokens=max_tokens)
                            meta_a = meta_g
                            provider = "gemini"
                            model_version = DEFAULT_GEMINI_MODEL
                        elif profile == "GEMINI_SEARCH_AUTO":
                        # use Gemini search wrapper
                            dbg("api_call_1_request", "Gemini (search auto) Anfrage")
                            mtoks = int(passA_search_tokens or max(max_tokens, 4000))
                            raw_text, meta_g = call_gemini_search_auto(Q, lang=lang, market=market, topn=topn,
                                                                       temperature=0.2, max_tokens=mtoks)
                            meta_a = meta_g
                            provider = "gemini"
                            model_version = DEFAULT_GEMINI_MODEL
                            for c in (meta_a.get("citations") or []):
                                evidence_list.append({
                                    "title": c.get("title"),
                                    "url": c.get("url"),
                                    "domain": c.get("domain"),
                                    "snippet": c.get("snippet", ""),
                                    "discovered_at": now_iso()
                                })
                        else:
                            raw_text = f"[ERROR] Unknown profile {profile}"
                            meta_a = {"status": "error"}

                        ans_excerpt = (raw_text or "")[:1500]
                        dbg("api_call_1_response",
                            "Pass A Antwort" if meta_a.get("status") == "completed" else "Fehler in Pass A",
                            meta={"status": meta_a.get("status"), "answer_excerpt": ans_excerpt})
                    
                    except Exception as ex:
                        raw_text = f"[ERROR Pass A: {ex}]"
                        meta_a = {"status": "error", "error": str(ex)}
                        dbg("api_call_1_response", "Fehler in Pass A", meta={"error": str(ex)})

                    # record raw answer row for all runs (success or error)
                    raw_resp_json = None
                    if isinstance(meta_a, dict):
                        raw_obj = meta_a.get("raw")
                        if isinstance(raw_obj, (dict, list)):
                            try:
                                raw_resp_json = json.dumps(raw_obj, ensure_ascii=False)
                            except Exception:
                                raw_resp_json = str(raw_obj)
                        elif raw_obj is not None:
                            raw_resp_json = str(raw_obj)

                    raw_rows.append({
                        "run_id": run_id,
                        "question_id": qid,
                        "profile": profile,
                        "language": lang,
                        "stakeholder": stake,
                        "answer_text": raw_text,
                        "status": meta_a.get("status", "unknown"),
                        "response_raw": raw_resp_json,
                        "ts": now_iso()
                    })

                    # handle cancellation after pass A
                    if check_cancel():
                        dbg("cancelled", "Abgebrochen")
                        break

                    # ---------- PASS B ----------
                    # format evidence list for coder prompt
                    cse_fmt = render_cse_items_for_coder(evidence_list)
                    user_prompt = (user_by_lang.get(lang) or user_by_lang.get("en") or "")
                    user_prompt = user_prompt.replace("{RAW_TEXT}", raw_text).replace("{CSE_ITEMS}", cse_fmt)

                    dbg("normalize_request", "Pass B Normalisierung")
                    try:
                        obj = pass_b_normalize(system_coder, user_prompt, model=MODEL_PASSB, max_tokens=5000)
                        dbg("normalize_response", "Pass B OK")
                    except Exception as ex:
                        dbg("normalize_response", "Pass B Fehler", meta={"error": str(ex)})
                        obj = {
                            "brand": brand,
                            "topic": topic,
                            "market": market,
                            "language": lang,
                            "profile": profile,
                            "intent": intent,
                            "variant": variant,
                            "inclusion": None,
                            "sentiment_label": "unknown",
                            "aspect_scores": {"visibility": None, "sentiment": None, "narrative": [], "risk_flags": [], "sources": []},
                            "evidence": [],
                            "freshness_index": 0.0,
                            "freshness_known_pct": 0.0,
                            "run_meta": {"run_timestamp": now_iso(), "provider": provider, "model_version": model_version, "error": str(ex)},
                            "confidence": "low"
                        }

                    # ---------- Enrichment ----------
                    dbg("enrich_start", "Enrichment der Evidence")
                    try:
                        obj["evidence"] = enrich_evidence(obj.get("evidence") or evidence_list, domain_seed_csv)
                        obj["freshness_index"] = freshness_index(obj["evidence"])
                        obj["freshness_known_pct"] = known_freshness_pct(obj["evidence"])
                        dbg("enrich_done", "Enrichment fertig")
                    except Exception as ex:
                        obj["evidence"] = obj.get("evidence") or []
                        obj["freshness_index"] = obj.get("freshness_index", 0.0)
                        obj["freshness_known_pct"] = obj.get("freshness_known_pct", 0.0)
                        obj.setdefault("run_meta", {})["enrichment_error"] = str(ex)
                        dbg("enrich_done", "Enrichment Fehler ignoriert", meta={"error": str(ex)})

                    # ---------- Meta and Row Assembly ----------
                    obj["brand"] = brand
                    obj["topic"] = topic
                    obj["market"] = market
                    obj["language"] = lang
                    obj["profile"] = profile
                    obj["intent"] = intent
                    obj["variant"] = variant
                    # add stakeholder into normalized object
                    obj["stakeholder"] = stake
                    obj.setdefault("run_meta", {}).update({
                        "run_timestamp": now_iso(),
                        "provider": provider,
                        "model_version": model_version,
                        "search_params": {"n": topn, "hl": lang, "gl": market} if profile == "GOOGLE_OVERVIEW" else {},
                        "pass_a_status": meta_a.get("status")
                    })

                    # record runs, norm and evidence rows
                    runs_rows.append({
                        "run_id": run_id,
                        "question_id": qid,
                        "category": row["category"],
                        "profile": profile,
                        "language": lang,
                        "stakeholder": stake,
                        "intent": intent,
                        "variant": variant,
                        "brand": brand,
                        "topic": topic,
                        "market": market,
                        "raw_text_len": len(raw_text or ""),
                        "ts": now_iso(),
                        "provider": provider,
                        "model": model_version
                    })
                    norm_rows.append({"run_id": run_id, **obj})
                    for e in obj.get("evidence") or []:
                        ev_rows.append({"run_id": run_id, **e})

                    done_jobs += 1
                    dbg("progress", f"{done_jobs}/{total_jobs} erledigt", meta={"done": done_jobs, "total": total_jobs})
                    # small pause to avoid overwhelming rate limits
                    time.sleep(0.12)

                # end replicate loop
            # end stakeholder loop
        # end profile loop
        # after finishing question
    # end question loop

    
def _sentiment_label_from_score(x: Optional[float]) -> str:
    if x is None: return "unknown"
    try:
        v = float(x)
    except Exception:
        return "unknown"
    if v <= -0.2: return "negative"
    if v >= 0.2:  return "positive"
    return "neutral"

def _avg_pairwise_jaccard(list_of_sets: List[Set[str]]) -> Optional[float]:
    n = len(list_of_sets)
    if n < 2: return None
    import itertools
    s = 0.0; k = 0
    for a,b in itertools.combinations(list_of_sets, 2):
        u = len(a.union(b)); j = 1.0 if u==0 else len(a.intersection(b))/u
        s += j; k += 1
    return s/k if k else None

QUALITY_FLAG_WEIGHTS: Dict[str, float] = {
    "HALLUCINATION_SUSPECTED": 1.0,
    "CONFUSES_WITH_OTHER_BRAND": 0.9,
    "OUTDATED_INFO": 0.7,
    "MISSING_KEY_ASPECTS": 0.6,
    "UNSUPPORTED_SUPERLATIVES": 0.5,
    "INCOHERENT_OR_OFF_TOPIC": 0.7,
    "SOURCE_BIAS_RISK": 0.5,
    "DATA_MISMATCH": 0.7,
    "REGION_CONTEXT_MISMATCH": 0.6,
    "OTHER_QUALITY_ISSUE": 0.3,
}
def _quality_indices(flags) -> (float, float):
    if not isinstance(flags, (list, tuple)): return 0.0, 1.0
    total = 0.0; seen = set()
    for f in flags:
        if not f: continue
        k = str(f).strip().upper()
        if k in seen: continue
        seen.add(k); total += float(QUALITY_FLAG_WEIGHTS.get(k, 0.3))
    risk = min(1.0, max(0.0, total)); score = 1.0 - risk
    return risk, score

def compute_stability_metrics(df_norm: pd.DataFrame, df_evi: pd.DataFrame) -> pd.DataFrame:
    if df_norm is None or df_norm.empty: return pd.DataFrame()
    df = df_norm.copy()
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = df.get("aspect_scores.sentiment").apply(_sentiment_label_from_score)
    keys = ["question_id","profile","language","stakeholder","category","intent","variant"]
    for k in keys:
        if k not in df.columns: df[k] = None
    domap = {}
    if df_evi is not None and not df_evi.empty and "run_id" in df_evi.columns:
        for rid, g in df_evi.groupby("run_id"):
            domap[rid] = set([str(x).lower() for x in g["domain"].dropna().tolist() if str(x).strip()])
    rows = []
    for grp, g in df.groupby(keys, dropna=False):
        runs = g["run_id"].tolist() if "run_id" in g.columns else []
        if len(runs) < 2: continue
        mode = g["sentiment_label"].mode(); top = mode.iloc[0] if not mode.empty else "unknown"
        agree = (g["sentiment_label"] == top).mean()
        def to_sets(col):
            vals = []
            if col in g.columns:
                for v in g[col].tolist():
                    if isinstance(v, list): vals.append(set([str(s).strip().lower() for s in v if str(s).strip()]))
                    else: vals.append(set())
            return vals
        narr_sets = to_sets("aspect_scores.narrative")
        risk_sets  = to_sets("aspect_scores.risk_flags")
        qual_sets  = to_sets("aspect_scores.quality_flags")
        narr_j = _avg_pairwise_jaccard(narr_sets)
        risk_j = _avg_pairwise_jaccard(risk_sets)
        qual_j = _avg_pairwise_jaccard(qual_sets)
        src_sets = [domap.get(rid, set()) for rid in runs]
        src_j = _avg_pairwise_jaccard(src_sets)
        risk_vals = []; score_vals = []
        if "aspect_scores.quality_flags" in g.columns:
            for flags in g["aspect_scores.quality_flags"].tolist():
                r,s = _quality_indices(flags); risk_vals.append(r); score_vals.append(s)
        avg_risk = sum(risk_vals)/len(risk_vals) if risk_vals else None
        avg_score = sum(score_vals)/len(score_vals) if score_vals else None
        d = dict(zip(keys, grp))
        d.update({
            "num_runs": len(runs),
            "agreement_rate_sentiment": round(float(agree),4) if agree is not None else None,
            "jaccard_narrative_avg": round(float(narr_j),4) if narr_j is not None else None,
            "jaccard_risk_flags_avg": round(float(risk_j),4) if risk_j is not None else None,
            "jaccard_quality_flags_avg": round(float(qual_j),4) if qual_j is not None else None,
            "jaccard_sources_avg": round(float(src_j),4) if src_j is not None else None,
            "top_sentiment_label": top,
            "avg_quality_risk_index": round(float(avg_risk),4) if avg_risk is not None else None,
            "avg_quality_score": round(float(avg_score),4) if avg_score is not None else None,
        })
        rows.append(d)
    return pd.DataFrame(rows)

# ---------- Export (pandas 3.0 keyword-only)
    df_runs = pd.DataFrame(runs_rows)
    df_norm = pd.json_normalize(norm_rows, max_level=2) if norm_rows else pd.DataFrame()
    df_evi  = pd.DataFrame(ev_rows)
    df_cfg  = pd.DataFrame(cfg_rows)
    df_raw  = pd.DataFrame(raw_rows)

    # Per-run quality indices on Normalized
    if not df_norm.empty and 'aspect_scores.quality_flags' in df_norm.columns:
        _risk=[]; _score=[]
        for flags in df_norm['aspect_scores.quality_flags'].tolist():
            r,s = _quality_indices(flags); _risk.append(r); _score.append(s)
        df_norm['quality_risk_index'] = _risk; df_norm['quality_score'] = _score

    # Stability metrics across runs
    df_stab = compute_stability_metrics(df_norm, df_evi)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
        if not df_runs.empty: df_runs.to_excel(excel_writer=xlw, sheet_name="Runs", index=False)
        if not df_norm.empty: df_norm.to_excel(excel_writer=xlw, sheet_name="Normalized", index=False)
        if not df_evi.empty:  df_evi.to_excel(excel_writer=xlw, sheet_name="Evidence", index=False)
        if not df_cfg.empty:  df_cfg.to_excel(excel_writer=xlw, sheet_name="Config", index=False)
        if not df_raw.empty:  df_raw.to_excel(excel_writer=xlw, sheet_name="RawAnswers", index=False)
        if 'df_stab' in locals() and not df_stab.empty: df_stab.to_excel(excel_writer=xlw, sheet_name="Stability_Metrics", index=False)

    return {"out": out_xlsx, "runs": len(df_runs), "norm": len(df_norm), "evi": len(df_evi)}