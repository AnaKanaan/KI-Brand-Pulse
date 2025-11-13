# ki_rep_monitor.py
import os, json, time, math, re, uuid, threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
import requests
import tldextract

# --------------------
# Models / Endpoints
# --------------------
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_CHAT  = os.getenv("MODEL_CHAT", "gpt-5-chat-latest")   # Pass A
MODEL_PASSB = os.getenv("MODEL_PASS_B", "gpt-5")             # Pass B (reasoning medium)

# --------------------
# Small utils
# --------------------
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
    vals = [math.exp(-float(e.get("age_days", 0))/90.0)
            for e in evidence if isinstance(e.get("age_days"), (int, float))]
    return sum(vals)/len(vals) if vals else 0.0

def known_freshness_pct(evidence: List[Dict[str, Any]]) -> float:
    return (sum(1 for e in evidence if e.get("age_days") is not None)/len(evidence)) if evidence else 0.0

# --------------------
# HTTP helpers
# --------------------
def openai_responses(payload: Dict[str, Any]) -> Dict[str, Any]:
    key = require_env_runtime("OPENAI_API_KEY")
    r = requests.post(
        f"{OPENAI_BASE}/responses",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120
    )
    # treat 2xx and 4xx/5xx consistently with error details
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")
    return r.json()

def gemini_generate_text(prompt: str, system: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Optional: Verwende Gemini für die Overview-Zusammenfassung, falls GEMINI_API_KEY gesetzt.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""
    mdl = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={api_key}"

    parts = []
    if system:
        parts.append({"text": f"[SYSTEM]\n{system}\n" })
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2}
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=60)
    if r.status_code >= 400:
        # Fallback: leere Antwort
        return ""
    d = r.json()
    try:
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""

# --------------------
# Responses API text/citations helpers
# --------------------
def extract_text_from_responses(data: Dict[str, Any]) -> str:
    if isinstance(data, dict) and data.get("output_text"):
        return data["output_text"]
    out = data.get("output") if isinstance(data, dict) else None
    try:
        return out[0]["content"][0]["text"]
    except Exception:
        return json.dumps(data)

def extract_text_and_citations(data: Dict[str, Any]):
    """
    Holt output_text + url_citation-Annos. Rückgabe: (text, citations[list])
    citation: {url,title,domain,start,end,snippet}
    """
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
    # Dedup nach URL -> längstes snippet behalten
    dedup = {}
    for c in citations:
        u = c["url"]
        if u not in dedup or len((c.get("snippet") or "")) > len((dedup[u].get("snippet") or "")):
            dedup[u] = c
    return text, list(dedup.values())

def response_status(data: Dict[str, Any]) -> str:
    try:
        return (data.get("status") or "unknown").lower()
    except Exception:
        return "unknown"

# --------------------
# Search & prompts
# --------------------
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
    d = r.json()
    items = d.get("items") or []
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
    data = openai_responses({
        "model": MODEL_CHAT,
        "input": [
            {"role": "system", "content": sys},
            {"role": "user",   "content": user}
        ],
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens)
    })
    return extract_text_from_responses(data)

def overview_substitute(query: str, cse_items: List[Dict[str, Any]], lang: str,
                        temperature: float = 0.2, max_tokens: int = 900) -> str:
    """
    Falls GEMINI_API_KEY gesetzt ist, nutze Gemini; sonst OpenAI.
    """
    sys = f"Antworte in {lang}. Nutze ausschließlich die bereitgestellten Treffer. Zitiere Domains in Klammern, z. B. (nzz.ch)."
    body = []
    for i, it in enumerate(cse_items, 1):
        domain = it.get("displayLink") or normalize_domain(it.get("link", ""))
        body.append(f"{i}) {it.get('title','')} — {it.get('link','')} — {domain} — {it.get('snippet','')}")
    combined = f"{sys}\n\nFrage:\n{query}\n\nTreffer:\n" + "\n".join(body)

    # Try Gemini if available
    if os.getenv("GEMINI_API_KEY"):
        txt = gemini_generate_text(prompt=combined)
        if txt:
            return txt
    # Fallback: OpenAI
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

# --------------------
# JSON normalizer (Pass B)
# --------------------
def load_coder_prompts(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pass_b_normalize(system_prompt: str, user_prompt: str, model: Optional[str] = None, max_tokens: int = 800) -> Dict[str, Any]:
    """
    Pass B: normalisiert in striktes JSON.
    Responses API: JSON-Erzwingung via text.format
    gpt-5: reasoning=medium
    """
    m = model or MODEL_PASSB
    payload = {
        "model": m,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_output_tokens": int(max_tokens),
        "reasoning": {"effort": "medium"},
        "text": {"format": "json"}   # <- korrekt statt response_format
    }
    data = openai_responses(payload)
    txt = extract_text_from_responses(data).strip()
    try:
        return json.loads(txt)
    except Exception:
        txt2 = re.sub(r"^```json|```$", "", txt, flags=re.M).strip()
        return json.loads(txt2)

# --------------------
# Evidence enrichment
# --------------------
def enrich_evidence(evidence: List[Dict[str, Any]], domain_seed_csv: str) -> List[Dict[str, Any]]:
    seeds = pd.read_csv(domain_seed_csv).fillna("")
    direct = {}
    for _, row in seeds.iterrows():
        dt = row["domain_type"]
        ex = str(row["example_domains"] or "")
        for d in ex.split(";"):
            d = d.strip()
            if d:
                direct[normalize_domain(d)] = dt

    rules = []
    for _, row in seeds.iterrows():
        dt = row["domain_type"]
        tlds = [t.strip() for t in str(row["tld_hints"] or "").split(";") if t.strip()]
        kws  = [k.strip().lower() for k in str(row["keyword_hints"] or "").split(";") if k.strip()]
        rules.append((dt, tlds, kws))

    now = datetime.utcnow()
    out = []
    for e in evidence:
        domain = normalize_domain(e.get("domain") or e.get("url") or "")
        dt = direct.get(domain)
        if not dt:
            for dtc, tlds, kws in rules:
                if any(domain.endswith(t) for t in tlds if t):
                    dt = dtc
                    break
            if not dt:
                text = (e.get("title", "") + " " + e.get("snippet", "")).lower()
                for dtc, tlds, kws in rules:
                    if any(k in text for k in kws if k):
                        dt = dtc
                        break
        if not dt:
            dt = "other"

        pub = e.get("published_at")
        discovered = e.get("discovered_at") or now_iso()
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
            "country": e.get("country") or "unknown",
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "discovered_at": discovered,
            "age_days": age_days,
            "freshness_bucket": freshness_bucket(age_days),
            "snippet": e.get("snippet", "")
        })
    return out

# --------------------
# Questions normalization
# --------------------
def _canonize_questions_df(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q.columns = [str(c).strip() for c in q.columns]
    lower = {c.lower(): c for c in q.columns}
    cmap = {}
    # tolerantes Mapping
    if "id" in lower:           cmap[lower["id"]] = "question_id"
    if "query" in lower:        cmap[lower["query"]] = "question_text"
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

# --------------------
# Pass A call helpers
# --------------------
def call_chat_no_search(q: str, temperature: float = 0.5, max_tokens: int = 900):
    data = openai_responses({
        "model": MODEL_CHAT,
        "input": [{"role": "user", "content": q}],
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens)
    })
    st = response_status(data)
    text = extract_text_from_responses(data)
    return text, {"status": st, "raw": data}

def call_chat_search_auto(q: str, temperature: float = 0.25, max_tokens: int = 1200):
    data = openai_responses({
        "model": MODEL_CHAT,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "input": [{"role": "user", "content": q}],
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens)
    })
    st = response_status(data)
    text, citations = extract_text_and_citations(data)
    return text, {"status": st, "citations": citations, "raw": data}

# --------------------
# Main pipeline
# --------------------
def run_pipeline(
    brand: str, topic: str, market: str,
    languages: List[str], profiles: List[str],
    question_xlsx: str, out_xlsx: str,
    domain_seed_csv: str, coder_prompts_json: str,
    topn: int = 5, num_runs: int = 3, categories: Optional[List[str]] = None,
    question_ids: Optional[List[int]] = None,
    comp1: str = "", comp2: str = "", comp3: str = "",
    temperature_chat_no: float = 0.5, temperature_chat_search: float = 0.25,
    max_tokens: int = 900, wrapper_mode: str = 'free_emulation',
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> Dict[str, Any]:

    def emit(ev: Dict[str, Any]):
        if progress:
            try:
                progress(ev)
            except Exception:
                pass

    if cancel_event and cancel_event.is_set():
        emit({"t": time.time(), "phase":"cancelled", "msg":"Abgebrochen vor Start"})
        return {"out": out_xlsx, "runs": 0, "norm": 0, "evi": 0}

    emit({"t": time.time(), "phase": "start", "msg": "Lese Questions"})
    q = pd.read_excel(question_xlsx, sheet_name="Questions")
    emit({"t": time.time(), "phase": "questions_loaded", "msg": f"{len(q)} Zeilen geladen"})
    q = _canonize_questions_df(q)

    orig_len = len(q)
    if languages:
        q = q[q["language"].isin(languages)].copy()
    if categories:
        q = q[q["category"].isin(categories)].copy()
    if question_ids:
        q = q[q["question_id"].isin(question_ids)].copy()

    emit({"t": time.time(), "phase": "filters_applied", "msg": f"Nach Filtern: {len(q)} / {orig_len} Fragen"})

    if q.empty:
        raise ValueError("Keine Fragen nach Filter übrig.")

    prompts      = load_coder_prompts(coder_prompts_json)
    system_coder = prompts["system_coder"]
    user_by_lang = prompts["user_coder"]

    runs_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []
    ev_rows:   List[Dict[str, Any]] = []
    raw_rows:  List[Dict[str, Any]] = []  # Originalantworten Pass A
    cfg_rows   = [
        {"key": "wrapper_mode", "value": wrapper_mode},
        {"key": "profiles", "value": ",".join(profiles)}
    ]

    total_jobs = len(q) * max(1, len(profiles)) * max(1, int(num_runs))
    done_jobs  = 0

    def check_cancel() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    for _, row in q.iterrows():
        if check_cancel():
            emit({"t": time.time(), "phase": "cancelled", "msg": "Abgebrochen"})
            break

        qid    = int(row["question_id"])
        intent = int(row["intent"])
        variant= int(row["variant"])
        lang   = row["language"]

        base_q = (
            str(row["question_text"])
            .replace("<BRAND>", brand).replace("<TOPIC>", topic).replace("<MARKET>", market)
            .replace("<COMP1>", comp1).replace("<COMP2>", comp2).replace("<COMP3>", comp3)
        )

        for profile in profiles:
            for r in range(int(num_runs)):
                if check_cancel():
                    emit({"t": time.time(), "phase": "cancelled", "msg": "Abgebrochen"})
                    break

                run_id = f"{qid}-{profile}-{lang}-r{r+1}-{uuid.uuid4().hex[:8]}"
                evidence_list: List[Dict[str, Any]] = []
                provider = "openai"
                model_version = MODEL_CHAT

                Q = base_q  # (Wrapper-Modi wären hier erweiterbar)
                emit({"t": time.time(), "phase": "build_prompt", "msg": f"Frage {qid} / {profile} / r{r+1} Prompt: {Q}"[:300]})

                # ---- PASS A
                try:
                    if profile == "CHATGPT_NO_SEARCH":
                        raw_text, meta_a = call_chat_no_search(Q, temperature=temperature_chat_no, max_tokens=max_tokens)
                    elif profile == "CHATGPT_SEARCH_AUTO":
                        # höheres Token-Budget gegen truncation
                        raw_text, meta_a = call_chat_search_auto(Q, temperature=temperature_chat_search, max_tokens=max(1200, int(max_tokens)))
                        # Citations in Evidence übernehmen
                        for c in (meta_a.get("citations") or []):
                            evidence_list.append({
                                "title": c.get("title"),
                                "url": c.get("url"),
                                "domain": c.get("domain"),
                                "snippet": c.get("snippet", ""),
                                "discovered_at": now_iso()
                            })
                    elif profile == "GOOGLE_OVERVIEW":
                        items = cse_list(base_q, lang, market, topn)
                        raw_text = overview_substitute(base_q, items, lang)
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
                    else:
                        raw_text = f"[ERROR] Unknown profile {profile}"
                        meta_a = {"status": "error"}

                    emit({"t": time.time(), "phase": "api_call_1_response",
                          "msg": "Pass A Antwort",
                          "meta": {"status": meta_a.get("status"),
                                   "prompt_excerpt": Q[:200],
                                   "answer_excerpt": (raw_text or "")[:300]}})
                except Exception as ex:
                    # Pass A Fehler – wir protokollieren und gehen weiter (Pass B bekommt Fehlerobjekt)
                    raw_text = f"[ERROR Pass A: {ex}]"
                    meta_a = {"status": "error", "error": str(ex)}

                # Originalantwort persistent
                raw_rows.append({
                    "run_id": run_id,
                    "question_id": qid,
                    "profile": profile,
                    "language": lang,
                    "answer_text": raw_text,
                    "status": meta_a.get("status", "unknown"),
                    "ts": now_iso()
                })

                if check_cancel():
                    emit({"t": time.time(), "phase": "cancelled", "msg": "Abgebrochen"})
                    break

                # ---- PASS B (Normalisierung)
                prompts = load_coder_prompts(coder_prompts_json)
                system_coder = prompts["system_coder"]
                user_by_lang = prompts["user_coder"]

                cse_fmt = render_cse_items_for_coder(evidence_list)
                user_prompt = (user_by_lang.get(lang) or user_by_lang.get("en") or "").replace("{RAW_TEXT}", raw_text).replace("{CSE_ITEMS}", cse_fmt)

                emit({"t": time.time(), "phase": "normalize_request", "msg": "Pass B Normalisierung"})

                try:
                    obj = pass_b_normalize(system_coder, user_prompt, model=MODEL_PASSB, max_tokens=800)
                except Exception as ex:
                    emit({"t": time.time(), "phase": "normalize_response", "msg": "Pass B Fehler", "meta": {"error": str(ex)}})
                    obj = {
                        "brand": brand, "topic": topic, "market": market,
                        "language": lang, "profile": profile, "intent": intent, "variant": variant,
                        "inclusion": None, "sentiment_label": "unknown",
                        "aspect_scores": {"visibility": None, "sentiment": None, "narrative": [], "risk_flags": [], "sources": []},
                        "evidence": [], "freshness_index": 0.0, "freshness_known_pct": 0.0,
                        "run_meta": {"run_timestamp": now_iso(), "provider": provider, "model_version": model_version, "error": str(ex)},
                        "confidence": "low"
                    }

                # Enrichment
                emit({"t": time.time(), "phase": "enrich_start", "msg": "Enrichment der Evidence"})
                try:
                    obj["evidence"] = enrich_evidence(obj.get("evidence") or evidence_list, domain_seed_csv)
                    obj["freshness_index"] = freshness_index(obj["evidence"])
                    obj["freshness_known_pct"] = known_freshness_pct(obj["evidence"])
                    emit({"t": time.time(), "phase": "enrich_done", "msg": "Enrichment fertig"})
                except Exception as ex:
                    obj["evidence"] = obj.get("evidence") or []
                    obj["freshness_index"] = obj.get("freshness_index", 0.0)
                    obj["freshness_known_pct"] = obj.get("freshness_known_pct", 0.0)
                    obj.setdefault("run_meta", {})["enrichment_error"] = str(ex)

                # Meta auffüllen
                obj["brand"] = brand; obj["topic"] = topic; obj["market"] = market
                obj["language"] = lang; obj["profile"] = profile; obj["intent"] = intent; obj["variant"] = variant
                obj.setdefault("run_meta", {}).update({
                    "run_timestamp": now_iso(),
                    "provider": provider,
                    "model_version": model_version,
                    "search_params": {"n": topn, "hl": lang, "gl": market} if profile == "GOOGLE_OVERVIEW" else {},
                    "pass_a_status": meta_a.get("status")
                })

                runs_rows.append({
                    "run_id": run_id,
                    "question_id": qid,
                    "category": row["category"],
                    "profile": profile,
                    "language": lang,
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
                for e in obj["evidence"]:
                    ev_rows.append({"run_id": run_id, **e})

                done_jobs += 1
                emit({"t": time.time(), "phase": "progress", "msg": f"{done_jobs}/{total_jobs} erledigt",
                      "meta": {"done": done_jobs, "total": total_jobs}})

                time.sleep(0.15)  # kleine Entzerrung

            if check_cancel():
                break

    # Output Excel
    df_runs = pd.DataFrame(runs_rows)
    df_norm = pd.json_normalize(norm_rows, max_level=2) if norm_rows else pd.DataFrame()
    df_evi  = pd.DataFrame(ev_rows)
    df_cfg  = pd.DataFrame(cfg_rows)
    df_raw  = pd.DataFrame(raw_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
        if not df_runs.empty: df_runs.to_excel(xlw, "Runs", index=False)
        if not df_norm.empty: df_norm.to_excel(xlw, "Normalized", index=False)
        if not df_evi.empty:  df_evi.to_excel(xlw, "Evidence", index=False)
        if not df_cfg.empty:  df_cfg.to_excel(xlw, "Config", index=False)
        if not df_raw.empty:  df_raw.to_excel(xlw, "RawAnswers", index=False)

    return {"out": out_xlsx, "runs": len(df_runs), "norm": len(df_norm), "evi": len(df_evi)}
