
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, math, argparse, re
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import requests
import tldextract

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")

MODEL_CHAT = "gpt-5-chat-latest"

def require_env(var):
    v = os.getenv(var)
    if not v:
        raise SystemExit(f"Environment variable {var} is required.")
    return v

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize_domain(url_or_domain: str) -> str:
    if not url_or_domain:
        return ""
    if "://" not in url_or_domain:
        host = url_or_domain.split("/")[0]
    else:
        try:
            host = requests.utils.urlparse(url_or_domain).hostname or url_or_domain
        except Exception:
            host = url_or_domain
    host = host.lower().lstrip("www.")
    ext = tldextract.extract(host)
    if not ext.suffix:
        return host
    return f"{ext.domain}.{ext.suffix}"

def freshness_bucket(age_days: int|None):
    if age_days is None: return "unknown"
    if age_days <= 0: return "today"
    if age_days <= 7: return "≤7d"
    if age_days <= 30: return "≤30d"
    if age_days <= 90: return "≤90d"
    if age_days <= 365: return "≤365d"
    return ">365d"

def freshness_index(evidence: List[Dict[str,Any]]):
    vals = []
    for e in evidence:
        ad = e.get("age_days")
        if isinstance(ad, (int,float)):
            vals.append(math.exp(-float(ad)/90.0))
    return sum(vals)/len(vals) if vals else 0.0

def known_freshness_pct(evidence):
    if not evidence: return 0.0
    known = sum(1 for e in evidence if e.get("age_days") is not None)
    return known/len(evidence)

def openai_responses(payload: Dict[str,Any]) -> Dict[str,Any]:
    require_env("OPENAI_API_KEY")
    hdr = { "Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json" }
    r = requests.post(f"{OPENAI_BASE}/responses", headers=hdr, data=json.dumps(payload), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")
    return r.json()

def extract_text_from_responses(data: Dict[str,Any]) -> str:
    try:
        if "output_text" in data and data["output_text"]:
            return data["output_text"]
        out = data.get("output") or []
        if out and out[0].get("content") and out[0]["content"][0].get("text"):
            return out[0]["content"][0]["text"]
    except Exception:
        pass
    return json.dumps(data)

def cse_list(q: str, lang: str, market: str, num: int = 5) -> List[Dict[str,Any]]:
    require_env("GOOGLE_API_KEY"); require_env("GOOGLE_CSE_ID")
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": q,
        "num": num,
        "hl": lang,
        "gl": market
    }
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"CSE HTTP {r.status_code}: {r.text}")
    data = r.json()
    items = data.get("items") or []
    out = []
    for it in items:
        out.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "snippet": it.get("snippet"),
            "displayLink": it.get("displayLink"),
            "htmlSnippet": it.get("htmlSnippet"),
            "pagemap": it.get("pagemap")
        })
    return out

def overview_substitute(query: str, cse_items: List[Dict[str,Any]], lang: str, temperature=0.2, max_tokens=900) -> str:
    sys_prompt = f"Antworte in {lang}. Nutze ausschließlich die bereitgestellten Treffer. Zitiere Domains im Fließtext (z. B. (nzz.ch))."
    lines = ["Frage:", query, "", "Treffer:"]
    for i, it in enumerate(cse_items, 1):
        domain = it.get("displayLink") or (requests.utils.urlparse(it.get("link","")).hostname or "")
        lines.append(f"{i}) {it.get('title')} — {it.get('link')} — {domain} — {it.get('snippet','')}")
    user_prompt = "\\n".join(lines)
    payload = {
        "model": MODEL_CHAT,
        "input": [
            {"role":"system","content": sys_prompt},
            {"role":"user","content": user_prompt}
        ],
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    data = openai_responses(payload)
    return extract_text_from_responses(data)

def call_chat_no_search(query: str, temperature=0.5, max_tokens=900) -> str:
    payload = {
        "model": MODEL_CHAT,
        "input": [{"role":"user","content": query}],
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    data = openai_responses(payload)
    return extract_text_from_responses(data)

def call_chat_search_auto(query: str, temperature=0.25, max_tokens=900) -> str:
    payload = {
        "model": MODEL_CHAT,
        "tools": [{"type":"web_search"}],
        "tool_choice": "auto",
        "input": [{"role":"user","content": query}],
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    data = openai_responses(payload)
    return extract_text_from_responses(data)

def load_coder_prompts(path: str) -> Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pass_b_normalize(system_prompt: str, user_prompt: str) -> Dict[str,Any]:
    payload = {
        "model": MODEL_CHAT,
        "input": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_prompt}
        ],
        "temperature": 0.0,
        "max_output_tokens": 900
    }
    data = openai_responses(payload)
    txt = extract_text_from_responses(data).strip()
    try:
        return json.loads(txt)
    except Exception:
        txt2 = re.sub(r"^```json|```$", "", txt).strip()
        return json.loads(txt2)

def enrich_evidence(evidence: List[Dict[str,Any]], domain_seed_csv: str) -> List[Dict[str,Any]]:
    seeds = pd.read_csv(domain_seed_csv).fillna("")
    direct = {}
    for _, row in seeds.iterrows():
        dt = row["domain_type"]; examples = str(row["example_domains"] or "")
        for d in examples.split(";"):
            d = d.strip()
            if d:
                direct[normalize_domain(d)] = dt
    rules = []
    for _, row in seeds.iterrows():
        dt = row["domain_type"]
        tld_list = [t.strip() for t in str(row["tld_hints"] or "").split(";") if t.strip()]
        kw_list = [k.strip().lower() for k in str(row["keyword_hints"] or "").split(";") if k.strip()]
        rules.append((dt, tld_list, kw_list))

    now = datetime.utcnow()
    out = []
    for e in evidence:
        domain = normalize_domain(e.get("domain") or e.get("url") or "")
        dt = direct.get(domain)
        if not dt:
            for dtc, tlds, kws in rules:
                if any(domain.endswith(t) for t in tlds if t):
                    dt = dtc; break
            if not dt:
                text = (e.get("title","") + " " + e.get("snippet","")).lower()
                for dtc, tlds, kws in rules:
                    if any(k in text for k in kws if k):
                        dt = dtc; break
        if not dt: dt = "other"

        pub = e.get("published_at")
        discovered = e.get("discovered_at") or now_iso()
        try:
            pub_dt = datetime.fromisoformat(pub.replace("Z","+00:00")) if pub else None
        except Exception:
            pub_dt = None
        age_days = (now - pub_dt).days if pub_dt else None
        out.append({
            "title": e.get("title"),
            "url": e.get("url"),
            "domain": domain,
            "domain_type": dt,
            "country": e.get("country") or "unknown",
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "discovered_at": discovered,
            "age_days": age_days,
            "freshness_bucket": freshness_bucket(age_days)
        })
    return out

def render_cse_items_for_coder(items: List[Dict[str,Any]]):
    lines = []
    for i, e in enumerate(items, 1):
        domain = e.get("domain") or e.get("displayLink") or ""
        lines.append(f"{i}) {e.get('title','')} — {e.get('url', e.get('link',''))} — {domain}")
    return "\\n".join(lines) if lines else "(keine)"

def run_pipeline(brand: str, topic: str, market: str, languages: List[str], profiles: List[str],
                 question_xlsx: str, out_xlsx: str, domain_seed_csv: str, coder_prompts_json: str,
                 topn: int = 5, num_runs: int = 3):
    q = pd.read_excel(question_xlsx, sheet_name="Questions")
    q = q[q["language"].isin(languages)].copy()

    prompts = load_coder_prompts(coder_prompts_json)
    system_coder = prompts["system_coder"]
    user_by_lang = prompts["user_coder"]

    runs_rows, norm_rows, ev_rows = [], [], []

    for _, row in q.iterrows():
        qid = int(row["question_id"]); intent = int(row["intent"]); variant = int(row["variant"]); lang = row["language"]
        qtext = str(row["question_text"]).replace("<BRAND>", brand).replace("<TOPIC>", topic).replace("<MARKET>", market)

        for profile in profiles:
            for r in range(num_runs):
                run_id = f"{qid}-{profile}-{lang}-r{r+1}-{int(time.time()*1000)}"
                raw_text = ""
                evidence_list = []
                provider = "openai"
                model_version = MODEL_CHAT

                if profile == "CHATGPT_NO_SEARCH":
                    raw_text = call_chat_no_search(qtext)
                elif profile == "CHATGPT_SEARCH_AUTO":
                    raw_text = call_chat_search_auto(qtext)
                elif profile == "GOOGLE_OVERVIEW":
                    items = cse_list(qtext, lang, market, topn)
                    raw_text = overview_substitute(qtext, items, lang)
                    for it in items:
                        evidence_list.append({
                            "title": it.get("title"),
                            "url": it.get("link"),
                            "domain": it.get("displayLink") or normalize_domain(it.get("link","")),
                            "discovered_at": now_iso()
                        })
                    provider = "google"; model_version = "overview-substitute"
                else:
                    raise ValueError(f"Unknown profile: {profile}")

                # Pass B normalize
                cse_fmt = render_cse_items_for_coder(evidence_list)
                user_prompt = (user_by_lang.get(lang) or user_by_lang["en"])\\
                    .replace("{RAW_TEXT}", raw_text)\\
                    .replace("{CSE_ITEMS}", cse_fmt)\\
                    .replace("{LANG}", lang)\\
                    .replace("{MARKET}", market)
                obj = pass_b_normalize(system_coder, user_prompt)

                # Deterministic enrichment
                obj["evidence"] = enrich_evidence(obj.get("evidence") or evidence_list, domain_seed_csv)
                obj["freshness_index"] = freshness_index(obj["evidence"])
                obj["freshness_known_pct"] = known_freshness_pct(obj["evidence"])

                # Meta
                obj["brand"] = brand
                obj["topic"] = topic
                obj["market"] = market
                obj["language"] = lang
                obj["profile"] = profile
                obj["intent"] = intent
                obj["variant"] = variant
                obj["run_meta"] = obj.get("run_meta") or {
                    "run_timestamp": now_iso(), "provider": provider, "model_version": model_version,
                    "search_params": {"n": topn, "hl": lang, "gl": market} if profile=="GOOGLE_OVERVIEW" else {}
                }

                runs_rows.append({
                    "run_id": run_id, "question_id": qid, "profile": profile, "language": lang,
                    "intent": intent, "variant": variant, "brand": brand, "topic": topic, "market": market,
                    "raw_text_len": len(raw_text), "ts": now_iso(), "provider": provider, "model": model_version
                })
                norm_rows.append({"run_id": run_id, **obj})
                for e in obj["evidence"]:
                    ev_rows.append({"run_id": run_id, **e})

                time.sleep(0.2)

    df_runs = pd.DataFrame(runs_rows)
    df_norm = pd.json_normalize(norm_rows, max_level=2)
    df_evi  = pd.DataFrame(ev_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
        df_runs.to_excel(xlw, sheet_name="Runs", index=False)
        df_norm.to_excel(xlw, sheet_name="Normalized", index=False)
        df_evi.to_excel(xlw, sheet_name="Evidence", index=False)

    print(f"Saved -> {out_xlsx}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--market", required=True, help="DE, CH, AT, etc.")
    ap.add_argument("--profiles", nargs="+", default=["CHATGPT_NO_SEARCH","CHATGPT_SEARCH_AUTO","GOOGLE_OVERVIEW"])
    ap.add_argument("--languages", nargs="+", default=["de"])
    ap.add_argument("--question-xlsx", default="ki_question_library.xlsx")
    ap.add_argument("--domain-seed-csv", default="domain_type_seed.csv")
    ap.add_argument("--coder-prompts-json", default="coder_prompts_passB.json")
    ap.add_argument("--out-xlsx", default="out.xlsx")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--num-runs", type=int, default=3)
    args = ap.parse_args()

    if "GOOGLE_OVERVIEW" in args.profiles:
        require_env("GOOGLE_API_KEY"); require_env("GOOGLE_CSE_ID")
    if "CHATGPT_NO_SEARCH" in args.profiles or "CHATGPT_SEARCH_AUTO" in args.profiles:
        require_env("OPENAI_API_KEY")

    run_pipeline(
        brand=args.brand, topic=args.topic, market=args.market,
        languages=args.languages, profiles=args.profiles,
        question_xlsx=args.question_xlsx, out_xlsx=args.out_xlsx,
        domain_seed_csv=args.domain_seed_csv, coder_prompts_json=args.coder_prompts_json,
        topn=args.topn, num_runs=args.num_runs
    )

if __name__ == "__main__":
    main()
