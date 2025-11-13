# streamlit_app.py
import os, time, json, threading, queue
import pandas as pd
import streamlit as st

from ki_rep_monitor import run_pipeline

st.set_page_config(page_title='KI-Reputation Monitor', layout='wide')
st.title('🔎 KI-Reputation Monitor — Final3')

# -------------------------
# Session Init
# -------------------------
if "runner" not in st.session_state:
    st.session_state.runner = {
        "thread": None,
        "cancel": None,
        "events": queue.Queue(),  # worker -> UI
        "last_event_ts": 0.0,
        "jobs_done": 0,
        "jobs_total": 0,
        "start_t": 0.0
    }

# -------------------------
# API Keys (Session only)
# -------------------------
with st.expander('🔐 API-Keys (nur Session, keine Speicherung)'):
    openai_key = st.text_input('OpenAI API Key', type='password', placeholder='sk-...')
    google_key = st.text_input('Google API Key', type='password', placeholder='AIza...')
    google_cx  = st.text_input('Google CSE ID (cx)', type='password', placeholder='custom search engine id')
    gemini_key = st.text_input('Gemini API Key', type='password', placeholder='AIza... (PaLM/Gemini key)')
    if st.button('Apply Keys'):
        if openai_key: os.environ['OPENAI_API_KEY'] = openai_key
        if google_key: os.environ['GOOGLE_API_KEY'] = google_key
        if google_cx:  os.environ['GOOGLE_CSE_ID']  = google_cx
        if gemini_key: os.environ['GEMINI_API_KEY'] = gemini_key
        st.success('Keys gesetzt (nur Session).')

# -------------------------
# Sidebar Config
# -------------------------
with st.sidebar:
    st.markdown("### Konfiguration")
    brand = st.text_input('Brand', 'DAK')
    topic = st.text_input('Topic', 'KI im Gesundheitswesen')
    market = st.text_input('Market', 'DE')
    comp1 = st.text_input('Competitor 1', 'TK')
    comp2 = st.text_input('Competitor 2', 'AOK')
    comp3 = st.text_input('Competitor 3', '')

    profiles = st.multiselect('Profiles',
                              ['CHATGPT_NO_SEARCH', 'CHATGPT_SEARCH_AUTO', 'GOOGLE_OVERVIEW'],
                              default=['CHATGPT_NO_SEARCH', 'CHATGPT_SEARCH_AUTO', 'GOOGLE_OVERVIEW'])

    languages = st.multiselect('Languages', ['de', 'fr', 'it', 'rm', 'en'], default=['de'])

    categories = st.multiselect('Categories',
                                ['BRANDED','UNBRANDED','THOUGHT_LEADERSHIP','RISK','BENCHMARK'],
                                default=['BRANDED','BENCHMARK','RISK'])

    question_ids_raw = st.text_input('Question IDs (comma-separated)', '')

    topn = st.number_input('Google CSE Top-N', 1, 10, 5)
    num_runs = st.number_input('Replicates per question', 1, 10, 1)
    temp_no = st.slider('Temp (Chat no search)', 0.0, 1.2, 0.5, 0.05)
    temp_auto = st.slider('Temp (Chat auto-search)', 0.0, 1.2, 0.25, 0.05)
    max_tokens = st.number_input('max_output_tokens (Pass A)', 100, 4096, 900, 50)
    wrapper_mode = st.selectbox('Pass-A Wrapper', ['free_emulation','stabilized'], index=0)

    st.markdown("### Model Settings")
    model_chat = st.text_input('Model (Pass A: Antwort)', os.getenv("MODEL_CHAT", "gpt-5-chat-latest"))
    model_passb = st.text_input('Model (Pass B: Codierung)', os.getenv("MODEL_PASS_B", "gpt-5"))

    # Apply models into env (session)
    if st.button('Apply Model Settings'):
        os.environ["MODEL_CHAT"] = model_chat.strip()
        os.environ["MODEL_PASS_B"] = model_passb.strip()
        st.success("Modelle gesetzt.")

    uploaded = st.file_uploader('Question Library (xlsx, optional)', type=['xlsx'])
    if uploaded is not None:
        question_xlsx = uploaded
    else:
        question_xlsx = 'ki_question_library.xlsx'

    st.divider()
    run_btn = st.button('🚀 Run')
    stop_btn = st.button('🛑 Stop')

# -------------------------
# Helpers
# -------------------------
def parse_ids(s: str):
    if not s or not s.strip():
        return None
    out = []
    for p in s.split(','):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out or None

def event_sink(ev: dict):
    """
    Worker -> UI queue
    """
    st.session_state.runner["events"].put(ev)

def eta_text(done: int, total: int, start_t: float) -> str:
    if done <= 0:
        return "–"
    elapsed = max(0.001, time.time() - start_t)
    rate = done / elapsed  # jobs per sec
    remaining = max(0, total - done)
    if rate <= 0:
        return "–"
    sec_left = int(remaining / rate)
    mins = sec_left // 60
    secs = sec_left % 60
    return f"{mins:02d}:{secs:02d}"

def save_uploaded_file_to_tmp(uploaded_file) -> str:
    if isinstance(uploaded_file, str):
        return uploaded_file
    path = f"/tmp/_qlib_{int(time.time())}.xlsx"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# -------------------------
# Runner lifecycle
# -------------------------
def start_worker():
    if not os.getenv('OPENAI_API_KEY'):
        st.error('Bitte zuerst OpenAI API Key setzen.')
        return

    # ensure queue empty
    q: queue.Queue = st.session_state.runner["events"]
    while not q.empty():
        try: q.get_nowait()
        except Exception: break

    # persist uploaded file
    q_path = save_uploaded_file_to_tmp(question_xlsx)

    # quick debug preview of Questions columns (safe)
    try:
        cols = list(pd.read_excel(q_path, sheet_name="Questions").columns)
        st.sidebar.write("Questions sheet columns:", cols)
    except Exception as ex:
        st.sidebar.write("Questions sheet columns: <Fehler beim Lesen>", str(ex))

    out_name = f'out_{int(time.time())}.xlsx'
    cancel_event = threading.Event()
    st.session_state.runner["cancel"] = cancel_event
    st.session_state.runner["start_t"] = time.time()
    st.session_state.runner["jobs_done"] = 0
    st.session_state.runner["jobs_total"] = 0

    def _progress(ev: dict):
        # Pick out progress
        meta = ev.get("meta") or {}
        if ev.get("phase") == "progress":
            done = int(meta.get("done", 0))
            total = int(meta.get("total", 0))
            st.session_state.runner["jobs_done"] = done
            st.session_state.runner["jobs_total"] = total
        event_sink(ev)

    def _work():
        try:
            res = run_pipeline(
                brand=brand, topic=topic, market=market,
                languages=languages, profiles=profiles,
                question_xlsx=q_path, out_xlsx=out_name,
                domain_seed_csv='domain_type_seed.csv',
                coder_prompts_json='coder_prompts_passB.json',
                topn=int(topn), num_runs=int(num_runs),
                categories=categories, question_ids=parse_ids(question_ids_raw),
                comp1=comp1, comp2=comp2, comp3=comp3,
                temperature_chat_no=float(temp_no),
                temperature_chat_search=float(temp_auto),
                max_tokens=int(max_tokens), wrapper_mode=wrapper_mode,
                progress=_progress, cancel_event=cancel_event
            )
            event_sink({"t": time.time(), "phase": "done", "msg": json.dumps(res)})
        except Exception as ex:
            event_sink({"t": time.time(), "phase": "error", "msg": str(ex)})

    th = threading.Thread(target=_work, name="runner-thread", daemon=True)
    st.session_state.runner["thread"] = th
    th.start()

def stop_worker():
    cancel_event = st.session_state.runner.get("cancel")
    if cancel_event:
        cancel_event.set()

# Buttons
if run_btn and (st.session_state.runner["thread"] is None or not st.session_state.runner["thread"].is_alive()):
    start_worker()

if stop_btn:
    stop_worker()

# -------------------------
# Live Panel (progress + debug)
# -------------------------
is_running = bool(st.session_state.runner.get("thread") and st.session_state.runner["thread"].is_alive())
jobs_done  = st.session_state.runner.get("jobs_done", 0)
jobs_total = st.session_state.runner.get("jobs_total", 0)
start_t    = st.session_state.runner.get("start_t", time.time())

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Status**")
    st.write("Running..." if is_running else "Idle")
with c2:
    st.markdown("**Fortschritt**")
    st.write(f"{jobs_done}/{jobs_total}" if jobs_total else "–")
with c3:
    st.markdown("**ETA**")
    st.write(eta_text(jobs_done, jobs_total, start_t))

progress_bar = st.progress(0.0)
if jobs_total > 0:
    progress_bar.progress(min(1.0, jobs_done / jobs_total))
else:
    progress_bar.progress(0.0)

# Debug/Event Stream
st.markdown("### 🪵 Debug-Protokoll")
log_box = st.container()
with log_box:
    # drain queue (bounded loop to avoid blocking)
    drained = 0
    while drained < 500 and not st.session_state.runner["events"].empty():
        ev = st.session_state.runner["events"].get()
        drained += 1
        phase = ev.get("phase", "")
        msg   = ev.get("msg", "")
        meta  = ev.get("meta") or {}
        tstr  = time.strftime("%H:%M:%S", time.localtime(ev.get("t", time.time())))
        lines = [f"{tstr} {phase} — {msg}"]
        # einige Zusatzinfos formatiert
        if meta:
            # kompakte Anzeige der Citations (falls vorhanden)
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
            # Sonstiges Debug kompakt ausgeben
            pe = meta.get("prompt_excerpt")
            if pe: lines.append(f"    prompt: {pe}")
            ae = meta.get("answer_excerpt")
            if ae: lines.append(f"    answer: {ae}")
        st.code("\n".join(lines))

# -------------------------
# If finished: load output, show KPIs
# -------------------------
if not is_running:
    # Versuche "out_*.xlsx" zu finden (der Worker hat den Dateinamen in 'done' geloggt)
    # Einfachster Weg: die neueste out_*.xlsx in CWD laden, falls vorhanden
    out_file = None
    try:
        # Worker hat "done" mit res{"out": "..."} geschickt – wir könnten das log lesen,
        # aber hier simpler: suche letzte out_*.xlsx im Arbeitsverzeichnis
        # (Streamlit Cloud: CWD ist Repo-Root)
        candidates = [f for f in os.listdir(".") if f.startswith("out_") and f.endswith(".xlsx")]
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
        try:
            inc = norm['inclusion'].fillna(False).astype(bool).mean()
        except Exception:
            inc = 0.0
        try:
            sent = norm['aspect_scores.sentiment'].astype(float).fillna(0).mean()
        except Exception:
            sent = 0.0
        try:
            fresh = norm.get('freshness_index', pd.Series([0]*len(norm))).astype(float).fillna(0).mean()
        except Exception:
            fresh = 0.0
        evid_by_run = (evid.groupby('run_id').size() if not evid.empty else pd.Series(dtype=int))
        ev_rate = (evid_by_run.gt(0).mean() if not evid_by_run.empty else 0.0)
        dom_div = (evid['domain'].nunique() if not evid.empty and 'domain' in evid.columns else 0)

        lbl_counts = norm.get('sentiment_label', pd.Series([])).value_counts() if not norm.empty else pd.Series([])
        pos_share = (lbl_counts.get('positive',0) / max(lbl_counts.sum(),1)) if not lbl_counts.empty else 0.0

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
                fb = evid['freshness_bucket'].value_counts().reindex(['today','≤7d','≤30d','≤90d','≤365d','>365d','unknown']).fillna(0).reset_index()
                fb.columns = ['bucket','count']
                st.bar_chart(fb.set_index('bucket'))
            else:
                st.info('Keine Evidence-Daten.')

        st.markdown('### Profile × Language — Inclusion Rate')
        try:
            inc_pf = norm.assign(incl=norm['inclusion'].fillna(False).astype(bool)).groupby(['profile','language'])['incl'].mean().reset_index()
            inc_pf['incl'] = (inc_pf['incl']*100).round(1)
            inc_pf = inc_pf.pivot(index='profile', columns='language', values='incl').fillna(0)
            st.bar_chart(inc_pf)
        except Exception:
            st.info('Nicht genug Daten für Profile×Language.')

        st.markdown('### Sentiment by Profile')
        try:
            s_pf = norm.groupby('profile')['aspect_scores.sentiment'].mean().reset_index().set_index('profile')
            st.bar_chart(s_pf)
        except Exception:
            pass

        st.subheader('Runs')
        st.dataframe(runs, use_container_width=True, hide_index=True)
        st.subheader('Evidence')
        st.dataframe(evid, use_container_width=True, hide_index=True)
        st.subheader('Normalized (flattened JSON)')
        st.dataframe(norm, use_container_width=True, hide_index=True)
        if not raw.empty:
            st.subheader('Raw Answers (Pass A)')
            st.dataframe(raw, use_container_width=True, hide_index=True)
        st.subheader('Config')
        if not cfg.empty:
            st.table(cfg)
    else:
        st.info('Keys setzen (optional), konfigurieren und **Run** starten.')
else:
    st.info('Lauf aktiv — Logs & Fortschritt siehe oben. Zum Abbrechen: **Stop**.')
