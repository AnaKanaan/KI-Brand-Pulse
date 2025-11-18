# streamlit_app.py
import os, time, json, threading, queue, pathlib
import pandas as pd

# ---- Auto-refresh helper (no external dependency) ----
def ___AUTOREFRESH___():
    import time as _t
    r = st.session_state.get('runner', {}) if isinstance(st.session_state.get('runner', {}), dict) else {}
    last = r.get('last_auto_refresh', 0.0)
    now  = _t.time()
    # throttle to at most ~1 rerun / 1.5s
    if now - last > 1.5:
        r['last_auto_refresh'] = now
        st.session_state.runner = r
        st.rerun()
# ------------------------------------------------------
import streamlit as st

from ki_rep_monitor import run_pipeline

st.set_page_config(page_title='KI-Reputation Monitor', layout='wide')
st.title('🔎 KI-Reputation Monitor — Final3')
st.markdown(
    """
    <div style='margin: 0.8rem 0; font-size: 1.05rem;'>
    <strong>„Was erzählt die KI‑Landschaft aktuell über meine Marke / mein Thema – und auf welcher Wissensbasis?“</strong><br>
    <em>„Wir messen, wie KI‑Assistenten deinen Ruf formen: welche Antworten Nutzer*innen bekommen, welche Quellen dahinterstehen und wie stabil diese Bilder sind.“</em>
    </div>
    """,
    unsafe_allow_html=True
)
with st.expander("ℹ️ Definitionen & Legende", expanded=False):
    st.markdown("""
**Profile (konzeptionell)**  
- **NO_SEARCH** (z. B. `CHATGPT_NO_SEARCH`, `GEMINI_NO_SEARCH`): Antwort ohne Websuche, auf Basis von Modellwissen. *Citations* sind leer (`[]`) bzw. nur aus Modellwissen.  
- **SEARCH_AUTO** (z. B. `CHATGPT_SEARCH_AUTO`, `GEMINI_SEARCH_AUTO`): LLM führt Suche aus (Google/Browse), extrahiert **Evidence** (Titel, URL, Domain, Datum/Snippet) und begründet Antworten.  
- **GOOGLE_OVERVIEW**: kompakter Überblick mit Quellenlisten („Top‑Treffer“) für schnelle Orientierung.

**KPIs & Scores**  
- **Alter (`age_days`)**: Differenz in Tagen zwischen `published_at` und jetzt.  
- **Freshness‑Bucket**: `today` / `≤7d` / `≤30d` / `≤90d` / `≤365d` / `>365d`.  
- **Freshness‑Index**: \\(\\text{avg}\\big(e^{-(\\text{age\\_days}/90)}\\big)\\) über alle Evidenzen (0..1; höher = aktueller).  
- **Sentiment‑Score**: kontinuierlich in \\([-1, +1]\\).  
- **Sentiment‑Label**: Schwellen −0.2 / +0.2 ⇒ *negativ* / *neutral* / *positiv*.  
- **Visibility**: \\([0,1]\\) – „Wie prominent steht die Marke im Antworttext im Fokus?“  
- **Inclusion**: boolesch – ob die Antwort gemäß Codierung **auf Thema/Marke einzahlt** (Accepted) oder nicht (Rejected).

**Hinweise zur Erhebung**  
- *Visibility, Sentiment* werden in **Pass B** explizit im Prompt definiert und von der LLM codiert.  
- *Alter, Freshness* werden **nachträglich** aus den Evidenz‑Feldern berechnet.  
- *Inclusion* folgt der Klassifikation (Accepted/Rejected) aus der Codierung.
""")

# =========================================================
# Session init (nur im UI-Thread)
# =========================================================
if "runner" not in st.session_state:
    st.session_state.runner = {
        "thread": None,
        "cancel": None,
        "events": queue.Queue(),    # worker -> UI
        "jobs_done": 0,
        "jobs_total": 0,
        "start_t": 0.0,
        "log_path": None,           # NDJSON live-log
    }

# =========================================================
# API Keys (Session only)
# =========================================================
with st.expander('🔐 API-Keys (nur Session, keine Speicherung)'):
    openai_key = st.text_input('OpenAI API Key', type='password', placeholder='sk-...')
    google_key = st.text_input('Google API Key', type='password', placeholder='AIza...')
    google_cx  = st.text_input('Google CSE ID (cx)', type='password', placeholder='custom search engine id')
    gemini_key = st.text_input('Gemini API Key', type='password', placeholder='AIza...')
    if st.button('Apply Keys'):
        if openai_key: os.environ['OPENAI_API_KEY'] = openai_key
        if google_key: os.environ['GOOGLE_API_KEY'] = google_key
        if google_cx:  os.environ['GOOGLE_CSE_ID']  = google_cx
        if gemini_key: os.environ['GEMINI_API_KEY'] = gemini_key
        st.success('Keys gesetzt (nur Session).')

# =========================================================
# Sidebar Config
# =========================================================
with st.sidebar:
    st.markdown("### Konfiguration")
    st.caption("**Wrapper‑Modi:**\n\n- **free_emulation**: das LLM antwortet frei ohne Zusatzrahmen.\n- **stabilized**: strengere Anweisungen & Formatvorgaben, konsistentere Struktur (z. B. JSON), Quellenhinweise, knappe Antworten.")
    brand  = st.text_input('Brand', 'DAK')
    topic  = st.text_input('Topic', 'KI im Gesundheitswesen')
    market = st.text_input('Market', 'DE')
    comp1  = st.text_input('Competitor 1', 'TK')
    comp2  = st.text_input('Competitor 2', 'AOK')
    comp3  = st.text_input('Competitor 3', '')

    # Available interaction profiles.  Gemini profiles are newly added for free chat and search support.
    profiles = st.multiselect(
        'Profiles',
        ['CHATGPT_NO_SEARCH', 'CHATGPT_SEARCH_AUTO', 'GOOGLE_OVERVIEW', 'GEMINI_NO_SEARCH', 'GEMINI_SEARCH_AUTO'],
        default=['CHATGPT_NO_SEARCH', 'CHATGPT_SEARCH_AUTO', 'GOOGLE_OVERVIEW']
    )

    languages = st.multiselect('Languages', ['de','fr','it','rm','en'], default=['de'])

    # Stakeholder perspective selection.  If none selected a generic perspective is used.
    stakeholders = st.multiselect(
        'Stakeholders',
        ['generic', 'Bewerber', 'Investor', 'Mitarbeitender', 'Endkonsument', 'Business-Kunde', 'Business-Partner', 'Provider', 'Entscheidungsträger aus Politik und Verwaltung'],
        default=['generic']
    )

    categories = st.multiselect('Categories',
                                ['BRANDED','UNBRANDED','THOUGHT_LEADERSHIP','RISK','BENCHMARK'],
                                default=['BRANDED','BENCHMARK','RISK'])

    question_ids_raw = st.text_input('Question IDs (comma-separated)', '')

    topn       = st.number_input('Google CSE Top-N', 1, 10, 5)
    num_runs   = st.number_input('Replicates per question', 1, 10, 1)
    # Temperature controls removed from UI; fixed per policy
    temp_no = 0.0
    temp_auto = 0.0
    # Increase default token limits for Pass A to accommodate longer responses
    max_tokens = 4000
    max_tokens_search = 4000
    wrapper_mode = st.selectbox('Pass-A Wrapper', ['free_emulation','stabilized'], index=0)


    uploaded = st.file_uploader('Question Library (xlsx, optional)', type=['xlsx'])
    question_xlsx = uploaded if uploaded is not None else 'ki_question_library.xlsx'

    st.markdown("### Debug & Limits")
    debug_level = st.selectbox("Debug-Detailgrad", ["verbose","basic","none"], index=0)
    max_questions = st.number_input("Max. Fragen (Testlauf, 0 = alle)", 0, 2000, 0, 10)

    st.divider()
    run_btn  = st.button('🚀 Run')
    stop_btn = st.button('🛑 Stop')
    clear_btn= st.button('🧹 Clear Logs')

# =========================================================
# Helpers (UI-Thread)
# =========================================================
def parse_ids(s: str):
    if not s or not s.strip(): return None
    out = []
    for p in s.split(','):
        p = p.strip()
        if not p: continue
        try: out.append(int(p))
        except: pass
    return out or None

def ui_event_sink(ev: dict):
    """Nur im UI-Thread st.session_state anfassen."""
    st.session_state.runner["events"].put(ev)
    lp = st.session_state.runner.get("log_path")
    if lp:
        try:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            pass

def eta_text(done: int, total: int, start_t: float) -> str:
    if done <= 0 or total <= 0: return "–"
    elapsed = max(0.001, time.time() - start_t)
    rate = done / elapsed
    if rate <= 0: return "–"
    remain = total - done
    sec_left = int(remain / rate)
    return f"{sec_left // 60:02d}:{sec_left % 60:02d}"

def save_uploaded_file_to_tmp(uploaded_file) -> str:
    if isinstance(uploaded_file, str):
        return uploaded_file
    path = f"/tmp/_qlib_{int(time.time())}.xlsx"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# =========================================================
# Runner (Worker darf KEIN st.* benutzen)
# =========================================================
def start_worker():
    if not os.getenv('OPENAI_API_KEY'):
        st.error('Bitte zuerst OpenAI API Key setzen.')
        return

    # Queue leeren
    q: queue.Queue = st.session_state.runner["events"]
    while not q.empty():
        try: q.get_nowait()
        except: break

    q_path = save_uploaded_file_to_tmp(question_xlsx)
    # schnelle Spaltenvorschau
    try:
        cols = list(pd.read_excel(q_path, sheet_name="Questions").columns)
        st.sidebar.write("Questions sheet columns:", cols)
        show_preview = st.sidebar.checkbox("Fragen‑Vorschau (bis 5 Zeilen)", value=False)
        if show_preview:
            try:
                df_prev = pd.read_excel(q_path, sheet_name="Questions").head(5)
            except Exception:
                xl = pd.ExcelFile(q_path)
                sheet = next((s for s in xl.sheet_names if s in ["de","en","fr","it","rm"]), xl.sheet_names[0])
                df_prev = pd.read_excel(q_path, sheet_name=sheet).head(5)
            st.sidebar.dataframe(df_prev, use_container_width=True, hide_index=True)
    except Exception as ex:
        st.sidebar.write("Questions sheet columns: <Fehler>", str(ex))

    out_name = f'out_{int(time.time())}.xlsx'
    st.session_state.runner['expected_xlsx'] = out_name
    cancel_event = threading.Event()
    st.session_state.runner["cancel"] = cancel_event
    st.session_state.runner["start_t"] = time.time()
    st.session_state.runner["jobs_done"] = 0
    st.session_state.runner["jobs_total"] = 0

    log_path = f"/tmp/_run_{int(time.time())}.ndjson"
    st.session_state.runner["log_path"] = log_path
    try:
        pathlib.Path(log_path).write_text("", encoding="utf-8")
    except Exception:
        st.session_state.runner["log_path"] = None
        log_path = None

    # Worker-lokale Sinks (keine session_state Zugriffe!)
    def worker_event_sink(ev: dict):
        try:
            q.put(ev)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _progress(ev: dict):
        # nur in Queue schieben, UI aktualisiert Counters beim Drain
        worker_event_sink(ev)

    def _work():
        try:
            res = run_pipeline(
                brand=brand, topic=topic, market=market,
                languages=languages, profiles=profiles,
                question_xlsx=q_path, out_xlsx=out_name,
                domain_seed_csv='domain_type_prompt.csv',
                coder_prompts_json='coder_prompts_passB.json',
                topn=int(topn), num_runs=int(num_runs),
                categories=categories, question_ids=parse_ids(question_ids_raw),
                comp1=comp1, comp2=comp2, comp3=comp3,
                temperature_chat_no=float(temp_no), temperature_chat_search=float(temp_auto),
                max_tokens=int(max_tokens), wrapper_mode=wrapper_mode,
                progress=_progress, cancel_event=cancel_event,
                debug_level=debug_level, max_questions=int(max_questions),
                passA_search_tokens=int(max_tokens_search),
                stakeholders=stakeholders
            )
            worker_event_sink({"t": time.time(), "phase": "done", "msg": json.dumps(res)})
        except Exception as ex:
            worker_event_sink({"t": time.time(), "phase": "error", "msg": str(ex)})

    st.session_state.runner["status"] = "starting"
    th = threading.Thread(target=_work, name="runner-thread", daemon=True)
    st.session_state.runner["thread"] = th
    th.start()
    st.session_state.runner["status"] = "running"

def stop_worker():
    c = st.session_state.runner.get("cancel")
    if c: c.set()

def clear_logs():
    q: queue.Queue = st.session_state.runner["events"]
    while not q.empty():
        try: q.get_nowait()
        except: break
    st.session_state.runner["log_path"] = None

# Buttons
if run_btn and (st.session_state.runner["thread"] is None or not st.session_state.runner["thread"].is_alive()):
    start_worker()

if stop_btn:
    stop_worker()

if clear_btn:
    clear_logs()

# =========================================================
# Live-Status & Logs
try:
    r = st.session_state.get('runner', {}) if isinstance(st.session_state.get('runner', {}), dict) else {}
    need_refresh = False
    th = r.get('thread')
    if th is not None and hasattr(th, 'is_alive') and th.is_alive():
        need_refresh = True
    status = str(r.get('status','')).lower()
    if status in ('starting','running','queued','active','working'):
        need_refresh = True
    try:
        prog = float(r.get('progress', 0) or 0)
        total = float(r.get('total', 0) or 0)
        if total > 0 and prog < total:
            need_refresh = True
    except Exception:
        pass
    if need_refresh:
        ___AUTOREFRESH___()
except Exception:
    pass
    # 2) Status-based triggering
    status = str(r.get('status','')).lower()
    if status in ('starting','running','queued','active','working'):
        need_refresh = True
    # 3) Progress-based triggering
    try:
        prog = float(r.get('progress', 0) or 0)
        total = float(r.get('total', 0) or 0)
        if total > 0 and prog < total:
            need_refresh = True
    except Exception:
        pass
    if need_refresh:
        try:
            # streamlit_autorefresh not available; using st.rerun() throttle
            ___AUTOREFRESH___()
        except Exception:
            import time as _t
            last = r.get('last_auto_refresh', 0.0)
            now  = _t.time()
            if now - last > 1.5:
                r['last_auto_refresh'] = now
                st.session_state.runner = r
                st.rerun()
except Exception:
    pass

# =========================================================
is_running = bool(st.session_state.runner.get("thread") and st.session_state.runner["thread"].is_alive())
jobs_done  = st.session_state.runner.get("jobs_done", 0)
jobs_total = st.session_state.runner.get("jobs_total", 0)
start_t    = st.session_state.runner.get("start_t", time.time())
log_path   = st.session_state.runner.get("log_path")

st.divider()
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown("**Status**");      st.write("Läuft…" if is_running else "Idle")
with c2: st.markdown("**Fortschritt**"); st.write(f"{jobs_done}/{jobs_total}" if jobs_total else "–")
with c3: st.markdown("**ETA**");         st.write(eta_text(jobs_done, jobs_total, start_t))
with c4:
    if log_path and os.path.exists(log_path):
        st.download_button("📥 Debug-Logs (NDJSON)", data=open(log_path,"rb").read(), file_name=os.path.basename(log_path))

progress_bar = st.progress(min(1.0, jobs_done / jobs_total) if jobs_total else 0.0)

st.markdown('### 🪵 Debug-Protokoll (live)')
log_box = st.container()
with log_box:
    drained = 0
    while drained < 1000 and not st.session_state.runner["events"].empty():
        ev = st.session_state.runner["events"].get()
        drained += 1

        # Fortschritt im UI aktualisieren (hier ist Streamlit-Kontext vorhanden)
        if ev.get("phase") == "progress":
            meta = ev.get("meta") or {}
            st.session_state.runner["jobs_done"]  = int(meta.get("done", 0))
            st.session_state.runner["jobs_total"] = int(meta.get("total", 0))

        phase = ev.get("phase", "")
        msg   = ev.get("msg", "")
        meta  = ev.get("meta") or {}
        tstr  = time.strftime("%H:%M:%S", time.localtime(ev.get("t", time.time())))
        lines = [f"{tstr} {phase} — {msg}"]

        # Quellen hübsch anzeigen
        cits = meta.get("citations") or []
        if cits:
            lines.append("    Quellen:")
            for j, c in enumerate(cits, 1):
                dom = c.get("domain", "")
                tit = (c.get("title") or dom or "").strip()
                sn  = (c.get("snippet") or "").strip()
                url = c.get("url", "")
                if sn:
                    lines.append(f"      {j}. {tit} ({dom}) – {sn} – {url}")
                else:
                    lines.append(f"      {j}. {tit} ({dom}) – {url}")

        pe = meta.get("prompt_excerpt"); ae = meta.get("answer_excerpt")
        if pe: lines.append(f"    prompt: {pe}")
        if ae: lines.append(f"    answer: {ae}")

        st.code("\n".join(lines))

# =========================================================
# Ergebnis-Panel wenn fertig
# =========================================================
if not is_running:
    out_file = None
    try:
        if st.session_state.runner.get('expected_xlsx') and os.path.exists(st.session_state.runner['expected_xlsx']):
            out_file = st.session_state.runner['expected_xlsx']
        else:
            candidates = [f for f in os.listdir('.') if f.startswith('out_') and f.endswith('.xlsx')]
            if candidates:
                out_file = max(candidates, key=lambda p: os.path.getmtime(p))
    except Exception:
        pass

    if out_file and os.path.exists(out_file):
        st.success(f'Fertig: {out_file}')
        with open(out_file, "rb") as fh:
            st.download_button('📥 Download Excel', data=fh.read(), file_name=out_file)

        try:
            xls = pd.ExcelFile(out_file)
            runs = pd.read_excel(xls, 'Runs') if "Runs" in xls.sheet_names else pd.DataFrame()
            norm = pd.read_excel(xls, 'Normalized') if "Normalized" in xls.sheet_names else pd.DataFrame()
            evid = pd.read_excel(xls, 'Evidence') if "Evidence" in xls.sheet_names else pd.DataFrame()
            cfg  = pd.read_excel(xls, 'Config') if "Config" in xls.sheet_names else pd.DataFrame()
            raw  = pd.read_excel(xls, 'RawAnswers') if "RawAnswers" in xls.sheet_names else pd.DataFrame()
        except Exception as ex:
            st.error(f"Excel lesen fehlgeschlagen: {ex}")
            runs = norm = evid = cfg = raw = pd.DataFrame()

        st.subheader('📊 KPIs')
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        # robust gegen pandas 3.0
        def _bool_mean(series: pd.Series) -> float:
            if series is None or series.empty: return 0.0
            s = series.astype('boolean').fillna(False)
            return float(s.mean())

        def _num_mean(series: pd.Series) -> float:
            if series is None or series.empty: return 0.0
            return pd.to_numeric(series, errors='coerce').fillna(0).mean()

        inc   = _bool_mean(norm.get('inclusion', pd.Series(dtype='boolean')))
        sent  = _num_mean(norm.get('aspect_scores.sentiment', pd.Series(dtype='float')))
        fresh = _num_mean(norm.get('freshness_index', pd.Series(dtype='float')))

        evid_by_run = (evid.groupby('run_id').size() if not evid.empty else pd.Series(dtype=int))
        ev_rate = (evid_by_run.gt(0).mean() if not evid_by_run.empty else 0.0)
        dom_div = int(evid['domain'].nunique()) if not evid.empty and 'domain' in evid.columns else 0

        lbl_col = norm.get('sentiment_label', pd.Series(dtype='object'))
        lbl_counts = lbl_col.value_counts() if not lbl_col.empty else pd.Series(dtype='int')
        pos_share  = (float(lbl_counts.get('positive', 0)) / max(int(lbl_counts.sum()), 1)) if not lbl_counts.empty else 0.0

        col1.metric('Inclusion Rate', f'{inc*100:.1f}%')
        col2.metric('Sentiment Ø', f'{sent:+.2f}')
        col3.metric('Freshness Index', f'{fresh:.2f}')
        col4.metric('% Runs mit Belegen', f'{ev_rate*100:.1f}%')
        col5.metric('Domain-Diversität', f'{dom_div}')
        col6.metric('Positiv-Label Anteil', f'{pos_share*100:.1f}%')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Domain Types**')
            if not evid.empty and 'domain_type' in evid.columns:
                dtc = evid['domain_type'].value_counts().reset_index()
                dtc.columns = ['domain_type','count']
                st.bar_chart(dtc.set_index('domain_type'))
            else:
                st.info('Keine Evidence-Daten.')
        with c2:
            st.markdown('**Freshness Buckets**')
            if not evid.empty and 'freshness_bucket' in evid.columns:
                order = ['today','≤7d','≤30d','≤90d','≤365d','>365d','unknown']
                fb = evid['freshness_bucket'].value_counts().reindex(order).fillna(0).reset_index()
                fb.columns = ['bucket','count']
                st.bar_chart(fb.set_index('bucket'))
            else:
                st.info('Keine Evidence-Daten.')

        st.markdown('### Profile × Language — Inclusion Rate')
        try:
            inc_pf = norm.assign(incl=norm['inclusion'].astype('boolean').fillna(False)) \
                         .groupby(['profile','language'])['incl'].mean().reset_index()
            inc_pf['incl'] = (inc_pf['incl']*100).round(1)
            inc_pf = inc_pf.pivot(index='profile', columns='language', values='incl').fillna(0)
            st.bar_chart(inc_pf)
        except Exception:
            st.info('Nicht genug Daten für Profile×Language.')

        st.markdown('### Sentiment by Profile')
        try:
            s_pf = pd.to_numeric(norm['aspect_scores.sentiment'], errors='coerce') \
                       .groupby(norm['profile']).mean().to_frame('sentiment')
            st.bar_chart(s_pf)
        except Exception:
            pass

        st.subheader('Runs'); st.dataframe(runs, use_container_width=True, hide_index=True)
        st.subheader('Evidence'); st.dataframe(evid, use_container_width=True, hide_index=True)
        st.subheader('Normalized (flattened JSON)'); st.dataframe(norm, use_container_width=True, hide_index=True)
        if not raw.empty:
            st.subheader('Raw Answers (Pass A)'); st.dataframe(raw, use_container_width=True, hide_index=True)
        st.subheader('Config'); 
        if not cfg.empty: st.table(cfg)
    else:
        st.info('Keys setzen (optional), konfigurieren und **Run** starten.')
else:
    st.info('Lauf aktiv — Logs & Fortschritt siehe oben. Mit **Stop** kannst du den Run abbrechen.')