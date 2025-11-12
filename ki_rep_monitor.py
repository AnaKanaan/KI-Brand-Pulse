# (short header omitted for brevity) — full implementation is included
import os, json, time, math, re, uuid, pandas as pd, requests, tldextract
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from threading import Event

OPENAI_BASE = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")
MODEL_CHAT = "gpt-5-chat-latest"

def now_iso(): return datetime.now(timezone.utc).isoformat()
def require_env_runtime(var):
    v=os.getenv(var); 
    if not v: raise RuntimeError(f"{var} not set (use UI).")
    return v

def normalize_domain(d):
    if not d: return ""
    try:
        from urllib.parse import urlparse
        h = urlparse(d).hostname or d
    except Exception:
        h = d
    h = h.lower().lstrip("www.")
    ext = tldextract.extract(h)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else h

def freshness_bucket(age_days):
    if age_days is None: return "unknown"
    if age_days<=0: return "today"
    if age_days<=7: return "≤7d"
    if age_days<=30: return "≤30d"
    if age_days<=90: return "≤90d"
    if age_days<=365: return "≤365d"
    return ">365d"

def freshness_index(evidence):
    vals=[math.exp(-float(e.get("age_days",0))/90.0) for e in evidence if isinstance(e.get("age_days"),(int,float))]
    return sum(vals)/len(vals) if vals else 0.0

def known_freshness_pct(evidence):
    return (sum(1 for e in evidence if e.get("age_days") is not None)/len(evidence)) if evidence else 0.0

# --------------------------- Progress/Debug utilities ---------------------------

def _emit(progress: Optional[Callable[[Dict[str, Any]], None]], phase: str, msg: str = "", **meta):
    """Safe event emitter for UI progress/debug panels."""
    if progress is None:
        return
    ev = {"t": time.time(), "phase": phase, "msg": msg, "meta": meta}
    try:
        progress(ev)
    except Exception:
        pass  # never break the pipeline due to UI hooks

_SECRET_PAT = re.compile(r'(sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9\.\-_]+)', re.IGNORECASE)
def _redact(s: str, maxlen: int = 4000) -> str:
    """Redact secrets and trim long strings for safe debug output."""
    if not isinstance(s, str):
        s = str(s)
    s = _SECRET_PAT.sub("•••", s)
    if len(s) > maxlen:
        s = s[:maxlen] + f"\n…[truncated {len(s)-maxlen} chars]"
    return s

# --------------------------- API helpers (unchanged) ---------------------------

def openai_responses(payload):
    key=require_env_runtime("OPENAI_API_KEY")
    r=requests.post(
        f"{OPENAI_BASE}/responses",
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        data=json.dumps(payload),
        timeout=120
    )
    if r.status_code>=400: raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")
    return r.json()

def extract_text_from_responses(data):
    if isinstance(data,dict) and data.get("output_text"): return data["output_text"]
    out=data.get("output") if isinstance(data,dict) else None
    try:
        return out[0]["content"][0]["text"]
    except Exception:
        return json.dumps(data)

def cse_list(q,lang,market,num=5):
    key=require_env_runtime("GOOGLE_API_KEY"); cx=require_env_runtime("GOOGLE_CSE_ID")
    r=requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key":key,"cx":cx,"q":q,"num":num,"hl":lang,"gl":market},
        timeout=60
    )
    if r.status_code>=400: raise RuntimeError(f"CSE HTTP {r.status_code}: {r.text}")
    d=r.json(); items=d.get("items") or []
    out=[]
    for it in items:
        out.append({"title":it.get("title"),"link":it.get("link"),"snippet":it.get("snippet"),"displayLink":it.get("displayLink")})
    return out

def overview_substitute(query, cse_items, lang, temperature=0.2, max_tokens=900):
    sys=f"Antworte in {lang}. Nutze ausschließlich die bereitgestellten Treffer. Zitiere Domains (z. B. (nzz.ch))."
    lines=["Frage:",query,"","Treffer:"]
    for i,it in enumerate(cse_items,1):
        domain=it.get("displayLink") or normalize_domain(it.get("link",""))
        lines.append(f"{i}) {it.get('title')} — {it.get('link')} — {domain} — {it.get('snippet','')}")
    user="\n".join(lines)
    data=openai_responses({"model":MODEL_CHAT,"input":[{"role":"system","content":sys},{"role":"user","content":user}],"temperature":temperature,"max_output_tokens":max_tokens})
    return extract_text_from_responses(data)

def call_chat_no_search(q,temperature=0.5,max_tokens=900):
    data=openai_responses({"model":MODEL_CHAT,"input":[{"role":"user","content":q}],"temperature":temperature,"max_output_tokens":max_tokens})
    return extract_text_from_responses(data)

def call_chat_search_auto(q,temperature=0.25,max_tokens=900):
    data=openai_responses({"model":MODEL_CHAT,"tools":[{"type":"web_search"}],"tool_choice":"auto","input":[{"role":"user","content":q}],"temperature":temperature,"max_output_tokens":max_tokens})
    return extract_text_from_responses(data)

def _load_wrappers(path="prompts/pass_a_wrappers.json"):
    try:
        with open(path,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def load_coder_prompts(path):
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def pass_b_normalize(system_prompt,user_prompt):
    data=openai_responses({"model":MODEL_CHAT,"input":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],"temperature":0.0,"max_output_tokens":1200})
    txt=extract_text_from_responses(data).strip()
    try: return json.loads(txt)
    except Exception:
        txt2=re.sub(r"^```json|```$","",txt).strip()
        return json.loads(txt2)

def enrich_evidence(evidence, domain_seed_csv):
    seeds=pd.read_csv(domain_seed_csv).fillna("")
    direct={}
    for _,row in seeds.iterrows():
        dt=row["domain_type"]; ex=str(row["example_domains"] or "")
        for d in ex.split(";"):
            d=d.strip()
            if d: direct[normalize_domain(d)]=dt
    rules=[]
    for _,row in seeds.iterrows():
        dt=row["domain_type"]
        tlds=[t.strip() for t in str(row["tld_hints"] or "").split(";") if t.strip()]
        kws=[k.strip().lower() for k in str(row["keyword_hints"] or "").split(";") if k.strip()]
        rules.append((dt,tlds,kws))
    now=datetime.utcnow()
    out=[]
    for e in evidence:
        domain=normalize_domain(e.get("domain") or e.get("url") or "")
        dt=direct.get(domain)
        if not dt:
            for dtc,tlds,kws in rules:
                if any(domain.endswith(t) for t in tlds if t): dt=dtc; break
            if not dt:
                text=(e.get("title","")+" "+e.get("snippet","")).lower()
                for dtc,tlds,kws in rules:
                    if any(k in text for k in kws if k): dt=dtc; break
        if not dt: dt="other"
        pub=e.get("published_at"); discovered=e.get("discovered_at") or now.isoformat()
        try: 
            pub_dt=datetime.fromisoformat(pub.replace("Z","+00:00")) if pub else None
        except Exception: 
            pub_dt=None
        age_days=(now-pub_dt).days if pub_dt else None
        out.append({"title":e.get("title"),"url":e.get("url") or e.get("link"),
                    "domain":domain,"domain_type":dt,"country":e.get("country") or "unknown",
                    "published_at":pub_dt.isoformat() if pub_dt else None,
                    "discovered_at":discovered,"age_days":age_days,"freshness_bucket":freshness_bucket(age_days)})
    return out

def render_cse_items_for_coder(items):
    lines=[]
    for i,e in enumerate(items,1):
        domain=e.get("domain") or e.get("displayLink") or ""
        lines.append(f"{i}) {e.get('title','')} — {e.get('url', e.get('link',''))} — {domain}")
    return "\n".join(lines) if lines else "(keine)"

# --------------------------- tolerant Questions schema ---------------------------

def _canonize_questions_df(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q.columns = [str(c).strip() for c in q.columns]

    # Tolerantes Mapping: Template nutzt "id" und "query"
    lower = {c.lower(): c for c in q.columns}
    cmap = {}
    if "id" in lower: cmap[lower["id"]] = "question_id"
    if "query" in lower: cmap[lower["query"]] = "question_text"
    for want in ["language","category","intent","variant"]:
        if want not in q.columns and want in lower:
            cmap[lower[want]] = want
    q = q.rename(columns=cmap)

    required = ["question_id","question_text","language","category","intent","variant"]
    missing = [c for c in required if c not in q.columns]
    if missing:
        raise KeyError(f"Missing columns in 'Questions': {missing}. Present: {list(q.columns)}")

    # Typen säubern
    q["question_id"] = pd.to_numeric(q["question_id"], errors="coerce").astype("Int64")
    q["intent"] = pd.to_numeric(q["intent"], errors="coerce").astype("Int64")
    q["variant"] = pd.to_numeric(q["variant"], errors="coerce").astype("Int64")
    q = q.dropna(subset=["question_id","question_text","language"])
    return q

# --------------------------- main pipeline (extended) ---------------------------

def run_pipeline(
    brand, topic, market, languages, profiles, question_xlsx, out_xlsx, domain_seed_csv, coder_prompts_json,
    topn=5, num_runs=3, categories=None, question_ids=None, comp1="", comp2="", comp3="",
    temperature_chat_no=0.5, temperature_chat_search=0.25, max_tokens=900, wrapper_mode='free_emulation',
    # NEW (optional):
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    debug: bool = False,
    cancel: Optional[Event] = None
):
    _emit(progress, "start", "Lese Questions")
    q = pd.read_excel(question_xlsx, sheet_name="Questions")
    q = _canonize_questions_df(q)
    _emit(progress, "questions_loaded", f"{len(q)} Zeilen geladen", columns=list(q.columns))

    if languages:  q = q[q["language"].isin(languages)].copy()
    if categories: q = q[q["category"].isin(categories)]
    if question_ids: q = q[q["question_id"].isin(question_ids)]
    if q.empty: 
        raise ValueError("Keine Fragen nach Filter übrig.")

    prompts=load_coder_prompts(coder_prompts_json)
    system_coder=prompts["system_coder"]; user_by_lang=prompts["user_coder"]
    wrappers=_load_wrappers()

    runs_rows=[]; norm_rows=[]; ev_rows=[]; cfg_rows=[{"key":"wrapper_mode","value":wrapper_mode},{"key":"profiles","value":",".join(profiles)}]

    # Fortschritt vorberechnen (Fragen * Profile * Runs)
    total = int(len(q) * max(len(profiles),1) * max(num_runs,1))
    start_t = time.time()
    done = 0

    for _,row in q.iterrows():
        if cancel is not None and cancel.is_set():
            _emit(progress, "cancelled", "Lauf abgebrochen (vor Verarbeitung)")
            # teilweises Resultat trotzdem schreiben?
            # -> schreiben wir am Ende regulär, falls schon etwas gesammelt
            break

        qid=int(row["question_id"]); intent=int(row["intent"]); variant=int(row["variant"]); lang=row["language"]
        base_q=(str(row["question_text"]).replace("<BRAND>",brand).replace("<TOPIC>",topic).replace("<MARKET>",market)
                .replace("<COMP1>",comp1).replace("<COMP2>",comp2).replace("<COMP3>",comp3))

        for profile in profiles:
            for r in range(num_runs):
                if cancel is not None and cancel.is_set():
                    _emit(progress, "cancelled", "Lauf abgebrochen (im Loop)")
                    break

                run_id=f"{qid}-{profile}-{lang}-r{r+1}-{uuid.uuid4().hex[:8]}"
                evidence_list=[]; provider="openai"; model_version=MODEL_CHAT

                try:
                    tmpl=(wrappers["wrappers"][wrapper_mode].get(lang) if wrappers else "{QUESTION}")
                    Q=(tmpl or "{QUESTION}").replace("{QUESTION}",base_q).replace("{LANG}",lang)
                    _emit(progress, "build_prompt", f"Frage {qid} / {profile} / r{r+1}",
                          qid=qid, intent=intent, variant=variant, language=lang,
                          prompt_excerpt=_redact(Q if debug else base_q, 1200))

                    if profile=="CHATGPT_NO_SEARCH":
                        _emit(progress, "api_call_1_request", "LLM (no search) Anfrage", model="PRIMARY", temperature=temperature_chat_no, max_tokens=max_tokens)
                        raw_text=call_chat_no_search(Q, temperature=temperature_chat_no, max_tokens=max_tokens)
                        _emit(progress, "api_call_1_response", "LLM (no search) Antwort", raw=_redact(raw_text, 2000))

                    elif profile=="CHATGPT_SEARCH_AUTO":
                        _emit(progress, "api_call_1_request", "LLM (search auto) Anfrage", model="PRIMARY", temperature=temperature_chat_search, max_tokens=max_tokens)
                        raw_text=call_chat_search_auto(Q, temperature=temperature_chat_search, max_tokens=max_tokens)
                        _emit(progress, "api_call_1_response", "LLM (search auto) Antwort", raw=_redact(raw_text, 2000))

                    elif profile=="GOOGLE_OVERVIEW":
                        _emit(progress, "api_call_1_request", "Google CSE", model="overview-substitute", topn=topn, lang=lang, market=market)
                        items=cse_list(base_q, lang, market, topn)
                        raw_text=overview_substitute(base_q, items, lang)
                        _emit(progress, "api_call_1_response", "Google Overview-Substitute Antwort", raw=_redact(raw_text, 2000))
                        for it in items:
                            evidence_list.append({"title":it.get("title"),"url":it.get("link"),
                                                  "domain":it.get("displayLink") or normalize_domain(it.get("link","")),
                                                  "discovered_at":now_iso()})
                        provider="google"; model_version="overview-substitute"

                    else:
                        raw_text=f"[ERROR] Unknown profile {profile}"

                except Exception as ex:
                    raw_text=f"[ERROR Pass A: {ex}]"
                    _emit(progress, "api_call_1_response", "Fehler in Pass A", raw=_redact(str(ex), 1200))

                # Normalisierung (Pass B)
                cse_fmt=render_cse_items_for_coder(evidence_list)
                user_prompt=(user_by_lang.get(lang) or user_by_lang["en"]).replace("{RAW_TEXT}",raw_text).replace("{CSE_ITEMS}",cse_fmt)
                try:
                    _emit(progress, "normalize_request", "Pass B Normalisierung")
                    obj=pass_b_normalize(system_coder, user_prompt)
                    _emit(progress, "normalize_response", "Pass B fertig", parsed_preview=_redact(json.dumps(obj, ensure_ascii=False), 1200))
                except Exception as ex:
                    obj={"brand":brand,"topic":topic,"market":market,"language":lang,"profile":profile,"intent":intent,"variant":variant,
                         "inclusion":None,"sentiment_label":"unknown",
                         "aspect_scores":{"visibility":None,"sentiment":None,"narrative":[],"risk_flags":[],"sources":[]},
                         "evidence":[],"freshness_index":0.0,"freshness_known_pct":0.0,
                         "run_meta":{"run_timestamp":now_iso(),"provider":provider,"model_version":model_version,"error":str(ex)},
                         "confidence":"low"}
                    _emit(progress, "normalize_response", "Pass B Fehler", raw=_redact(str(ex), 1200))

                # Enrichment
                try:
                    _emit(progress, "enrich_start", "Enrichment der Evidence")
                    obj["evidence"]=enrich_evidence(obj.get("evidence") or evidence_list, domain_seed_csv)
                    obj["freshness_index"]=freshness_index(obj["evidence"]); obj["freshness_known_pct"]=known_freshness_pct(obj["evidence"])
                    _emit(progress, "enrich_done", "Enrichment fertig",
                          evidence_count=len(obj["evidence"]),
                          freshness_index=obj["freshness_index"],
                          freshness_known_pct=obj["freshness_known_pct"])
                except Exception as ex:
                    obj["evidence"]=obj.get("evidence") or []; obj["freshness_index"]=obj.get("freshness_index",0.0); obj["freshness_known_pct"]=obj.get("freshness_known_pct",0.0)
                    obj.setdefault("run_meta",{})["enrichment_error"]=str(ex)
                    _emit(progress, "enrich_done", "Enrichment Fehler", error=_redact(str(ex), 1200))

                # Meta setzen & Zeilen sammeln (unverändert)
                obj["brand"]=brand; obj["topic"]=topic; obj["market"]=market
                obj["language"]=lang; obj["profile"]=profile; obj["intent"]=intent; obj["variant"]=variant
                obj.setdefault("run_meta",{}).update({"run_timestamp":now_iso(),"provider":provider,"model_version":model_version,"search_params":{"n":topn,"hl":lang,"gl":market} if profile=="GOOGLE_OVERVIEW" else {}})

                runs_rows.append({"run_id":run_id,"question_id":qid,"category":row["category"],"profile":profile,"language":lang,"intent":intent,"variant":variant,"brand":brand,"topic":topic,"market":market,"raw_text_len":len((obj.get('run_meta') or {}).get('error','')),"ts":now_iso(),"provider":provider,"model":model_version})
                norm_rows.append({"run_id":run_id, **obj})
                for e in obj["evidence"]: ev_rows.append({"run_id":run_id, **e})

                # Fortschritt & ETA nach jedem Run
                done += 1
                elapsed = time.time() - start_t
                per_item = elapsed / max(done, 1)
                remain = max(total - done, 0)
                eta_s = int(per_item * remain)
                pct = int(5 + 95 * (done / max(total, 1)))
                _emit(progress, "progress", f"{done}/{total} erledigt", done=done, total=total, pct=pct, eta_s=eta_s)

                time.sleep(0.2)  # (beibehalten)

            if cancel is not None and cancel.is_set():
                break

    # XLSX schreiben (unverändert) + Events
    df_runs=pd.DataFrame(runs_rows); df_norm=pd.json_normalize(norm_rows, max_level=2); df_evi=pd.DataFrame(ev_rows); df_cfg=pd.DataFrame(cfg_rows)
    _emit(progress, "write_output", f"Schreibe Ergebnis: {out_xlsx}", sizes={"runs":len(df_runs),"norm":len(df_norm),"evi":len(df_evi)})
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
        df_runs.to_excel(xlw,"Runs",index=False); df_norm.to_excel(xlw,"Normalized",index=False); df_evi.to_excel(xlw,"Evidence",index=False); df_cfg.to_excel(xlw,"Config",index=False)
    _emit(progress, "done", "Pipeline fertig", path=out_xlsx)

    return {"out":out_xlsx,"runs":len(df_runs),"norm":len(df_norm),"evi":len(df_evi)}
