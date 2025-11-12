# ki_rep_monitor.py
# End-to-end pipeline: Pass A (Antwort/Websuche) + Pass B (Normalisierung) + Enrichment + XLSX
# - OpenAI Responses API (kein temperature/top_p bei GPT-5)
# - Pass B erzwingt JSON-Ausgabe + reasoning={"effort":"medium"}
# - Optional: Gemini-basierte Overview-Zusammenfassung für Google CSE-Treffer

import os, json, time, math, re, uuid, pandas as pd, requests, tldextract
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from threading import Event

# --------------------------- Konfig / Modelle ---------------------------
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Vorgaben:
MODEL_PASS_A = os.getenv("MODEL_PASS_A", "gpt-5-chat-latest")  # Chat/Antwort + Websuche
MODEL_PASS_B = os.getenv("MODEL_PASS_B", "gpt-5")              # Normalisierung/Codierung (reasoning=medium)

# Für diese Modellfamilien keine Sampling-Parameter senden
UNSUPPORTED_BY_FAMILY = ("gpt-5", "o1", "o3", "o4")

# --------------------------- Utils ---------------------------

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
    if age_days <= 0: return "today"
    if age_days <= 7: return "≤7d"
    if age_days <= 30: return "≤30d"
    if age_days <= 90: return "≤90d"
    if age_days <= 365: return "≤365d"
    return ">365d"

def freshness_index(evidence: List[Dict[str, Any]]) -> float:
    vals = [math.exp(-float(e.get("age_days", 0)) / 90.0)
            for e in evidence if isinstance(e.get("age_days"), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0

def known_freshness_pct(evidence: List[Dict[str, Any]]) -> float:
    return (sum(1 for e in evidence if e.get("age_days") is not None) / len(evidence)) if evidence else 0.0

# -------- Progress/Debug Events --------
_SECRET_PAT = re.compile(r'(sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9\.\-_]+)', re.IGNORECASE)

def _redact(s: Any, maxlen: int = 4000) -> str:
    if not isinstance(s, str): s = str(s)
    s = _SECRET_PAT.sub("•••", s)
    return s if len(s) <= maxlen else s[:maxlen] + f"\n…[truncated {len(s)-maxlen} chars]"

def _emit(progress: Optional[Callable[[Dict[str, Any]], None]], phase: str, msg: str = "", **meta):
    if progress is None: return
    ev = {"t": time.time(), "phase": phase, "msg": msg, "meta": meta}
    try:
        progress(ev)
    except Exception:
        pass  # UI-Fehler dürfen die Pipeline nicht killen

# --------------------------- OpenAI Responses API ---------------------------

def _strip_unsupported_params(payload: dict, model: str) -> dict:
    p = dict(payload)
    if any(model.startswith(pref) for pref in UNSUPPORTED_BY_FAMILY):
        for k in ("temperature", "top_p", "logprobs", "n"):
            p.pop(k, None)
    return p

def openai_responses(payload: dict) -> dict:
    key = require_env_runtime("OPENAI_API_KEY")
    model = payload.get("model", "")
    safe_payload = _strip_unsupported_params(payload, model)
    t0 = time.time()
    r = requests.post(
        f"{OPENAI_BASE}/responses",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(safe_payload),
        timeout=180
    )
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:800]}")
    if r.status_code >= 400 or ("error" in data):
        err = (data.get("error") or {}).get("message") or r.text
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {err}")
    if not (data.get("output_text") or data.get("output")):
        raise RuntimeError("OpenAI: Response ohne output/output_text.")
    data.setdefault("debug", {})["latency_s"] = round(time.time() - t0, 3)
    return data

def extract_text_from_responses(data: dict) -> str:
    if isinstance(data, dict) and data.get("output_text"):
        return data["output_text"]
    out = data.get("output") if isinstance(data, dict) else None
    try:
        return out[0]["content"][0]["text"]
    except Exception:
        return json.dumps(data)

# --------------------------- Search & Summaries ---------------------------

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
    d = r.json(); items = d.get("items") or []
    return [{"title": it.get("title"),
             "link": it.get("link"),
             "snippet": it.get("snippet"),
             "displayLink": it.get("displayLink")} for it in items]

def overview_substitute(query: str, cse_items: List[Dict[str, Any]], lang: str,
                        max_tokens: int = 900) -> str:
    sys = f"Antworte in {lang}. Nutze ausschließlich die bereitgestellten Treffer. Zitiere Domains in Klammern, z. B. (nzz.ch)."
    lines = ["Frage:", query, "", "Treffer:"]
    for i, it in enumerate(cse_items, 1):
        domain = it.get("displayLink") or normalize_domain(it.get("link", ""))
        lines.append(f"{i}) {it.get('title')} — {it.get('link')} — {domain} — {it.get('snippet','')}")
    user = "\n".join(lines)
    data = openai_responses({
        "model": MODEL_PASS_A,
        "input": [{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        "max_output_tokens": max_tokens
    })
    return extract_text_from_responses(data)

def overview_substitute_gemini(query: str, cse_items: List[Dict[str, Any]], lang: str) -> Optional[str]:
    """Verwende Gemini (google-genai), falls verfügbar; sonst None → Fallback auf OpenAI-Variante."""
    try:
        from google import genai
    except Exception:
        return None
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    client = genai.Client(api_key=key)
    lines = [f"Antwortsprache: {lang}", "Frage:", query, "", "Treffer:"]
    for i, it in enumerate(cse_items, 1):
        domain = it.get("displayLink") or normalize_domain(it.get("link", ""))
        lines.append(f"{i}) {it.get('title')} — {it.get('link')} — {domain} — {it.get('snippet','')}")
    prompt = "\n".join(lines)
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return getattr(resp, "text", None) or ""
    except Exception:
        return None

# --------------------------- Chat Helpers ---------------------------

def call_chat_no_search(q: str, max_tokens: int = 900) -> str:
    data = openai_responses({
        "model": MODEL_PASS_A,
        "input": [{"role": "user", "content": q}],
        "max_output_tokens": max_tokens
    })
    return extract_text_from_responses(data)

def call_chat_search_auto(q: str, max_tokens: int = 900) -> str:
    data = openai_responses({
        "model": MODEL_PASS_A,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "input": [{"role": "user", "content": q}],
        "max_output_tokens": max_tokens
    })
    return extract_text_from_responses(data)

# --------------------------- Wrappers / Prompts ---------------------------

def _load_wrappers(path="prompts/pass_a_wrappers.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_coder_prompts(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pass_b_normalize(system_prompt: str, user_prompt: str) -> dict:
    """Pass B: strukturierte Normalisierung — JSON erzwungen; reasoning=medium."""
    data = openai_responses({
        "model": MODEL_PASS_B,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_output_tokens": 1200,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "medium"}
    })
    txt = extract_text_from_responses(data).strip()
    return json.loads(txt)

# --------------------------- Enrichment ---------------------------

def enrich_evidence(evidence: List[Dict[str, Any]], domain_seed_csv: str) -> List[Dict[str, Any]]:
    seeds = pd.read_csv(domain_seed_csv).fillna("")
    direct = {}
    for _, row in seeds.iterrows():
        dt = row["domain_type"]; ex = str(row["example_domains"] or "")
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
                if any(domain.endswith(t) for t in tlds if t): dt = dtc; break
            if not dt:
                text = (e.get("title", "") + " " + e.get("snippet", "")).lower()
                for dtc, tlds, kws in rules:
                    if any(k in text for k in kws if k): dt = dtc; break
        if not dt: dt = "other"

        pub = e.get("published_at")
        discovered = e.get("discovered_at") or now.isoformat()
        try:
            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00")) if pub else None
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
            "freshness_bucket": freshness_bucket(age_days)
        })
    return out

def render_cse_items_for_coder(items: List[Dict[str, Any]]) -> str:
    lines = []
    for i, e in enumerate(items, 1):
        domain = e.get("domain") or e.get("displayLink") or ""
        lines.append(f"{i}) {e.get('title','')} — {e.get('url', e.get('link',''))} — {domain}")
    return "\n".join(lines) if lines else "(keine)"

# --------------------------- Questions Schema ---------------------------

def _canonize_questions_df(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q.columns = [str(c).strip() for c in q.columns]

    # tolerantes Mapping
    lower = {c.lower(): c for c in q.columns}
    cmap = {}
    if "id" in lower:    cmap[lower["id"]] = "question_id"
    if "query" in lower: cmap[lower["query"]] = "question_text"
    for want in ["language", "category", "intent", "variant"]:
        if want not in q.columns and want in lower:
            cmap[lower[want]] = want
    q = q.rename(columns=cmap)

    required = ["question_id", "question_text", "language", "category", "intent", "variant"]
    missing = [c for c in required if c not in q.columns]
    if missing:
        raise KeyError(f"Missing columns in 'Questions': {missing}. Present: {list(q.columns)}")

    # Typen/Dropna
    q["question_id"] = pd.to_numeric(q["question_id"], errors="coerce").astype("Int64")
    q["intent"]      = pd.to_numeric(q["intent"], errors="coerce").astype("Int64")
    q["variant"]     = pd.to_numeric(q["variant"], errors="coerce").astype("Int64")
    q = q.dropna(subset=["question_id", "question_text", "language"])

    # Normalisieren
    q["language"] = q["language"].astype(str).str.strip().str.lower()
    q["category"] = q["category"].astype(str).str.strip()

    return q

# --------------------------- Hauptpipeline ---------------------------

def run_pipeline(
    brand: str, topic: str, market: str, languages: list, profiles: list,
    question_xlsx: str, out_xlsx: str, domain_seed_csv: str, coder_prompts_json: str,
    topn: int = 5, num_runs: int = 3, categories: Optional[list] = None, question_ids: Optional[list] = None,
    comp1: str = "", comp2: str = "", comp3: str = "",
    max_tokens: int = 900, wrapper_mode: str = "free_emulation",
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    debug: bool = False, cancel: Optional[Event] = None,
    use_gemini_overview: bool = True
) -> Dict[str, Any]:

    _emit(progress, "start", "Lese Questions")
    q = pd.read_excel(question_xlsx, sheet_name="Questions")
    q = _canonize_questions_df(q)
    _emit(progress, "questions_loaded", f"{len(q)} Zeilen geladen", columns=list(q.columns))

    # ---- Case-insensitive Filter (hier, NICHT auf Modulebene!) ----
    orig_len = len(q)
    langs = None
    cats  = None

    if languages:
        langs = [str(l).strip().lower() for l in languages if str(l).strip()]
        q = q[q["language"].isin(langs)].copy()

    if categories:
        cats = {str(c).strip().upper() for c in categories if str(c).strip()}
        q = q[q["category"].astype(str).str.upper().isin(cats)].copy()

    if question_ids:
        q = q[q["question_id"].isin(question_ids)].copy()

    _emit(
        progress,
        "filters_applied",
        f"Nach Filtern: {len(q)} / {orig_len} Fragen",
        languages=langs,
        categories=(sorted(list(cats)) if cats else None),
        question_ids=question_ids
    )

    if q.empty:
        raise ValueError("Keine Fragen nach Filter übrig.")

    prompts = load_coder_prompts(coder_prompts_json)
    system_coder = prompts["system_coder"]; user_by_lang = prompts["user_coder"]
    wrappers = _load_wrappers()

    runs_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []
    ev_rows:   List[Dict[str, Any]] = []
    cfg_rows = [{"key": "wrapper_mode", "value": wrapper_mode},
                {"key": "profiles", "value": ",".join(profiles)}]

    total = int(len(q) * max(len(profiles), 1) * max(num_runs, 1))
    start_t = time.time(); done = 0

    for _, row in q.iterrows():
        if cancel is not None and cancel.is_set():
            _emit(progress, "cancelled", "Lauf abgebrochen (vor Verarbeitung)")
            break

        qid = int(row["question_id"]); intent = int(row["intent"]); variant = int(row["variant"]); lang = row["language"]
        base_q = (str(row["question_text"])
                  .replace("<BRAND>", brand).replace("<TOPIC>", topic).replace("<MARKET>", market)
                  .replace("<COMP1>", comp1).replace("<COMP2>", comp2).replace("<COMP3>", comp3))

        for profile in profiles:
            for r in range(num_runs):
                if cancel is not None and cancel.is_set():
                    _emit(progress, "cancelled", "Lauf abgebrochen (im Loop)")
                    break

                run_id = f"{qid}-{profile}-{lang}-r{r+1}-{uuid.uuid4().hex[:8]}"
                evidence_list: List[Dict[str, Any]] = []
                provider = "openai"; model_version = MODEL_PASS_A

                try:
                    tmpl = (wrappers["wrappers"][wrapper_mode].get(lang) if wrappers else "{QUESTION}")
                    Q = (tmpl or "{QUESTION}").replace("{QUESTION}", base_q).replace("{LANG}", lang)
                    _emit(progress, "build_prompt", f"Frage {qid} / {profile} / r{r+1}",
                          qid=qid, intent=intent, variant=variant, language=lang,
                          prompt_excerpt=_redact(Q if debug else base_q, 1200))

                    if profile == "CHATGPT_NO_SEARCH":
                        _emit(progress, "api_call_1_request", "LLM (no search) Anfrage", model=MODEL_PASS_A, max_tokens=max_tokens)
                        raw_text = call_chat_no_search(Q, max_tokens=max_tokens)
                        _emit(progress, "api_call_1_response", "LLM (no search) Antwort", raw=_redact(raw_text, 2000))

                    elif profile == "CHATGPT_SEARCH_AUTO":
                        _emit(progress, "api_call_1_request", "LLM (search auto) Anfrage", model=MODEL_PASS_A, max_tokens=max_tokens)
                        raw_text = call_chat_search_auto(Q, max_tokens=max_tokens)
                        _emit(progress, "api_call_1_response", "LLM (search auto) Antwort", raw=_redact(raw_text, 2000))

                    elif profile == "GOOGLE_OVERVIEW":
                        _emit(progress, "api_call_1_request", "Google CSE", model="overview-substitute", topn=topn, lang=lang, market=market)
                        items = cse_list(base_q, lang, market, topn)

                        raw_text = None
                        if use_gemini_overview and os.getenv("GEMINI_API_KEY"):
                            raw_text = overview_substitute_gemini(base_q, items, lang)
                        if not raw_text:
                            raw_text = overview_substitute(base_q, items, lang, max_tokens=max_tokens)

                        _emit(progress, "api_call_1_response", "Overview Antwort", raw=_redact(raw_text, 2000))
                        for it in items:
                            evidence_list.append({
                                "title": it.get("title"),
                                "url": it.get("link"),
                                "domain": it.get("displayLink") or normalize_domain(it.get("link", "")),
                                "discovered_at": now_iso()
                            })
                        provider = "google"; model_version = "overview-substitute"

                    else:
                        raw_text = f"[ERROR] Unknown profile {profile}"

                except Exception as ex:
                    raw_text = f"[ERROR Pass A: {ex}]"
                    _emit(progress, "api_call_1_response", "Fehler in Pass A", raw=_redact(str(ex), 1200))

                # Pass B: Normalisierung (reasoning=medium ist in pass_b_normalize festgelegt)
                cse_fmt = render_cse_items_for_coder(evidence_list)
                user_prompt = (user_by_lang.get(lang) or user_by_lang["en"]) \
                              .replace("{RAW_TEXT}", raw_text).replace("{CSE_ITEMS}", cse_fmt)
                try:
                    _emit(progress, "normalize_request", "Pass B Normalisierung", model=MODEL_PASS_B, reasoning="medium")
                    obj = pass_b_normalize(system_coder, user_prompt)
                    _emit(progress, "normalize_response", "Pass B fertig",
                          parsed_preview=_redact(json.dumps(obj, ensure_ascii=False), 1200))
                except Exception as ex:
                    obj = {"brand": brand, "topic": topic, "market": market, "language": lang,
                           "profile": profile, "intent": intent, "variant": variant,
                           "inclusion": None, "sentiment_label": "unknown",
                           "aspect_scores": {"visibility": None, "sentiment": None, "narrative": [], "risk_flags": [], "sources": []},
                           "evidence": [], "freshness_index": 0.0, "freshness_known_pct": 0.0,
                           "run_meta": {"run_timestamp": now_iso(), "provider": provider, "model_version": model_version, "error": str(ex)},
                           "confidence": "low"}
                    _emit(progress, "normalize_response", "Pass B Fehler", raw=_redact(str(ex), 1200))

                # Enrichment
                try:
                    _emit(progress, "enrich_start", "Enrichment der Evidence")
                    obj["evidence"] = enrich_evidence(obj.get("evidence") or evidence_list, domain_seed_csv)
                    obj["freshness_index"] = freshness_index(obj["evidence"])
                    obj["freshness_known_pct"] = known_freshness_pct(obj["evidence"])
                    _emit(progress, "enrich_done", "Enrichment fertig",
                          evidence_count=len(obj["evidence"]),
                          freshness_index=obj["freshness_index"],
                          freshness_known_pct=obj["freshness_known_pct"])
                except Exception as ex:
                    obj["evidence"] = obj.get("evidence") or []
                    obj["freshness_index"] = obj.get("freshness_index", 0.0)
                    obj["freshness_known_pct"] = obj.get("freshness_known_pct", 0.0)
                    obj.setdefault("run_meta", {})["enrichment_error"] = str(ex)
                    _emit(progress, "enrich_done", "Enrichment Fehler", error=_redact(str(ex), 1200))

                # Meta setzen & sammeln
                obj["brand"] = brand; obj["topic"] = topic; obj["market"] = market
                obj["language"] = lang; obj["profile"] = profile; obj["intent"] = intent; obj["variant"] = variant
                obj.setdefault("run_meta", {}).update({
                    "run_timestamp": now_iso(), "provider": provider, "model_version": model_version,
                    "search_params": {"n": topn, "hl": lang, "gl": market} if profile == "GOOGLE_OVERVIEW" else {}
                })

                runs_rows.append({"run_id": run_id, "question_id": qid, "category": row["category"],
                                  "profile": profile, "language": lang, "intent": intent, "variant": variant,
                                  "brand": brand, "topic": topic, "market": market,
                                  "raw_text_len": len(str(obj.get('run_meta', {}).get('error', raw_text or ""))),
                                  "ts": now_iso(), "provider": provider, "model": model_version})
                norm_rows.append({"run_id": run_id, **obj})
                for e in obj["evidence"]:
                    ev_rows.append({"run_id": run_id, **e})

                # Fortschritt / ETA
                done += 1
                elapsed = time.time() - start_t
                per_item = elapsed / max(done, 1)
                remain = max(total - done, 0)
                eta_s = int(per_item * remain)
                pct = int(5 + 95 * (done / max(total, 1)))
                _emit(progress, "progress", f"{done}/{total} erledigt",
                      done=done, total=total, pct=pct, eta_s=eta_s)

                time.sleep(0.2)

            if cancel is not None and cancel.is_set():
                break

    # XLSX schreiben
    df_runs = pd.DataFrame(runs_rows)
    df_norm = pd.json_normalize(norm_rows, max_level=2)
    df_evi  = pd.DataFrame(ev_rows)
    df_cfg  = pd.DataFrame(cfg_rows)
    _emit(progress, "write_output", f"Schreibe Ergebnis: {out_xlsx}",
          sizes={"runs": len(df_runs), "norm": len(df_norm), "evi": len(df_evi)})
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
        df_runs.to_excel(xlw, "Runs", index=False)
        df_norm.to_excel(xlw, "Normalized", index=False)
        df_evi.to_excel(xlw, "Evidence", index=False)
        df_cfg.to_excel(xlw, "Config", index=False)
    _emit(progress, "done", "Pipeline fertig", path=out_xlsx)

    return {"out": out_xlsx, "runs": len(df_runs), "norm": len(df_norm), "evi": len(df_evi)}
