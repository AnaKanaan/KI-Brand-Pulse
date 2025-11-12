import os, time, json, threading
import pandas as pd, streamlit as st
from ki_rep_monitor import run_pipeline

st.set_page_config(page_title='KI-Reputation Monitor', layout='wide')
st.title('🔎 KI-Reputation Monitor — Final3')

# ---- Session-Init für Runner + Debug-Log ----
if "runner" not in st.session_state:
    st.session_state.runner = {"thread": None, "cancel": None, "start": None, "run_id": 0}
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []

# Orphan-Cleanup (falls alter Thread schon tot ist)
rt = st.session_state.runner["thread"]
if rt and not rt.is_alive():
    st.session_state.runner.update({"thread": None, "cancel": None, "start": None})

with st.expander('🔐 API-Keys (nur Session, keine Speicherung)'):
    openai_key = st.text_input('OpenAI API Key', type='password', placeholder='sk-...')
    google_key = st.text_input('Google API Key', type='password', placeholder='AIza...')
    google_cx  = st.text_input('Google CSE ID (cx)', type='password', placeholder='custom search engine id')
    if st.button('Apply Keys'):
        if openai_key: os.environ['OPENAI_API_KEY'] = openai_key
        if google_key: os.environ['GOOGLE_API_KEY'] = google_key
        if google_cx:  os.environ['GOOGLE_CSE_ID']  = google_cx
        st.success('Keys gesetzt (nur Session).')

with st.sidebar:
    brand = st.text_input('Brand', 'DAK')
    topic = st.text_input('Topic', 'KI im Gesundheitswesen')
    market = st.text_input('Market', 'DE')
    comp1 = st.text_input('Competitor 1', 'TK')
    comp2 = st.text_input('Competitor 2', 'AOK')
    comp3 = st.text_input('Competitor 3', '')

    profiles = st.multiselect('Profiles', ['CHATGPT_NO_SEARCH','CHATGPT_SEARCH_AUTO','GOOGLE_OVERVIEW'],
                              default=['CHATGPT_NO_SEARCH','CHATGPT_SEARCH_AUTO','GOOGLE_OVERVIEW'])
    languages = st.multiselect('Languages', ['de','fr','it','rm','en'], default=['de'])
    categories = st.multiselect('Categories', ['BRANDED','UNBRANDED','THOUGHT_LEADERSHIP','RISK','BENCHMARK'],
                                default=['BRANDED','BENCHMARK','RISK'])
    question_ids_raw = st.text_input('Question IDs (comma-separated)', '')

    topn = st.number_input('Google CSE Top-N', 1, 10, 5)
    num_runs = st.number_input('Replicates per question', 1, 10, 3)
    temp_no = st.slider('Temp (Chat no search)', 0.0, 1.2, 0.5, 0.05)
    temp_auto = st.slider('Temp (Chat auto-search)', 0.0, 1.2, 0.25, 0.05)
    max_tokens = st.number_input('max_output_tokens', 100, 4000, 900, 50)
    wrapper_mode = st.selectbox('Pass-A Wrapper', ['free_emulation','stabilized'], index=0)

    # Debug-Optionen
    debug_mode = st.checkbox("Debug-Modus", value=False)
    show_raw = st.checkbox("Raw API-Payloads anzeigen (redigiert)", value=False) if debug_mode else False
    unsafe_no_redact = st.checkbox("⚠︎ Ohne Schwärzung anzeigen (Risiko!)", value=False) if debug_mode else False

    uploaded = st.file_uploader('Question Library (xlsx, optional)', type=['xlsx'])
    if uploaded is not None:
        question_xlsx = uploaded
    else:
        question_xlsx = 'ki_question_library.xlsx'

    # Start/Abbrechen Buttons in der Sidebar
    run_btn = st.button('🚀 Run')
    abort_btn = st.button('⛔ Abbrechen')

def parse_ids(s):
    if not s or not s.strip(): return None
    out=[]
    for p in s.split(','):
        p=p.strip()
        if not p: continue
        try: out.append(int(p))
        except: pass
    return out or None

# Abbrechen (falls ein Run aktiv ist)
if abort_btn and st.session_state.runner["cancel"] is not None:
    st.session_state.runner["cancel"].set()
    st.info("Abbruch angefordert – bitte kurz warten …")

if run_btn:
    if not os.getenv('OPENAI_API_KEY'):
        st.error('Bitte zuerst OpenAI API Key setzen.')
        st.stop()

    # Falls noch ein alter Run lebt: zuerst abbrechen
    th = st.session_state.runner["thread"]
    if th and th.is_alive() and st.session_state.runner["cancel"]:
        st.session_state.runner["cancel"].set()
        st.warning("Vorherigen Lauf abgebrochen …")
        th.join(timeout=2)

    out_name = f'out_{int(time.time())}.xlsx'

    # Question XLSX nach /tmp kopieren (falls Upload)
    if isinstance(question_xlsx, str):
        q_path = question_xlsx
    else:
        q_path = f'/tmp/_qlib_{int(time.time())}.xlsx'
        with open(q_path, 'wb') as f: f.write(question_xlsx.getbuffer())

    # Quick-Preview der Questions
    try:
        qdf = pd.read_excel(q_path, sheet_name="Questions")
        st.sidebar.write("Questions sheet columns:", list(qdf.columns))
        st.sidebar.write("Preview:", qdf.head(3))
    except Exception as e:
        st.sidebar.error(f"Kann 'Questions' nicht lesen: {e}")
        st.stop()

    # UI-Platzhalter für Live-Status
    status = st.status("Pipeline startet …", state="running")
    prog   = st.progress(0)
    eta_box = st.empty()
    step_box = st.empty()
    debug_expander = st.expander("🪵 Debug-Protokoll", expanded=False) if debug_mode else None
    debug_area = debug_expander.container() if debug_mode else None

    # Hilfsfunktionen
    def _fmt_eta(sec: int) -> str:
        m, s = divmod(max(sec, 0), 60)
        return f"{m:02d}:{s:02d}"

    # Progress-/Debug-Hook aus der Pipeline
    def progress_hook(ev: dict):
        # Optional: unredigiert anzeigen (bewusstes Risiko)
        if unsafe_no_redact and "meta" in ev:
            pass  # Pipeline liefert bereits redigiert; hier zeigen wir "as is", wenn gewünscht

        st.session_state.debug_log.append(ev)
        meta = ev.get("meta", {})

        if "pct" in meta:
            try:
                prog.progress(int(meta["pct"]))
            except Exception:
                pass
        if "eta_s" in meta:
            eta_box.markdown(f"**ETA:** {_fmt_eta(int(meta['eta_s']))} min")

        step_box.write(f"**{ev.get('phase','')}** — {ev.get('msg','')}")
        if debug_mode and debug_area is not None:
            last = st.session_state.debug_log[-12:]
            lines = []
            for e in last:
                ts = time.strftime('%H:%M:%S', time.localtime(e["t"]))
                phase = e.get("phase","")
                msg = e.get("msg","")
                lines.append(f"- {ts} **{phase}** — {msg}")
                m = e.get("meta", {})
                if show_raw and "raw" in m:
                    lines.append(f"    ```\n{m['raw']}\n    ```")
                elif "prompt_excerpt" in m:
                    lines.append(f"    Prompt: `{m['prompt_excerpt']}`")
                elif "parsed_preview" in m:
                    lines.append(f"    Parsed: `{m['parsed_preview']}`")
            debug_area.markdown("\n".join(lines) if lines else "_(noch keine Events)_")

    # Worker-Thread starten
    result = {"res": None, "err": None}
    cancel_event = threading.Event()

    def _worker():
        try:
            result["res"] = run_pipeline(
                brand=brand, topic=topic, market=market,
                languages=languages, profiles=profiles,
                question_xlsx=q_path, out_xlsx=out_name,
                domain_seed_csv='domain_type_seed.csv',
                coder_prompts_json='coder_prompts_passB.json',
                topn=int(topn), num_runs=int(num_runs),
                categories=categories, question_ids=parse_ids(question_ids_raw),
                comp1=comp1, comp2=comp2, comp3=comp3,
                temperature_chat_no=float(temp_no), temperature_chat_search=float(temp_auto),
                max_tokens=int(max_tokens), wrapper_mode=wrapper_mode,
                # NEU:
                progress=progress_hook, debug=debug_mode, cancel=cancel_event
            )
        except Exception as e:
            result["err"] = e

    t0 = time.time()
    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    st.session_state.runner.update({"thread": th, "cancel": cancel_event, "start": t0, "run_id": st.session_state.runner["run_id"] + 1})

    # Herzschlag, bis Worker fertig
    while th.is_alive():
        elapsed = int(time.time() - t0)
        m, s = divmod(elapsed, 60)
        status.update(label=f"Läuft … ({m:02d}:{s:02d} elapsed)", state="running")
        time.sleep(1)

    # Abschluss
    prog.progress(100)
    if cancel_event.is_set() and not result["res"]:
        status.update(label="Abgebrochen", state="error")
        st.warning("Lauf wurde abgebrochen.")
        st.stop()
    elif result["err"]:
        status.update(label="Fehlgeschlagen", state="error")
        st.exception(result["err"])
        st.stop()
    else:
        status.update(label="Fertig", state="complete")

    # ----- Auswertung & Anzeige (dein bestehender Block unverändert beibehalten) -----
    xls = pd.ExcelFile(out_name)
    runs = pd.read_excel(xls, 'Runs')
    norm = pd.read_excel(xls, 'Normalized')
    evid = pd.read_excel(xls, 'Evidence')
    cfg  = pd.read_excel(xls, 'Config')

    st.success(f'Fertig: {out_name}')
    st.download_button('📥 Download Excel', data=open(out_name,'rb').read(), file_name=out_name)

    st.subheader('📊 KPIs')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    try: inc = norm['inclusion'].fillna(False).astype(bool).mean()
    except: inc = 0.0
    try: sent = norm['aspect_scores.sentiment'].astype(float).fillna(0).mean()
    except: sent = 0.0
    fresh = norm.get('freshness_index', pd.Series([0]*len(norm))).astype(float).fillna(0).mean()
    evid_by_run = evid.groupby('run_id').size() if not evid.empty else pd.Series(dtype=int)
    ev_rate = (evid_by_run.gt(0).mean() if not evid_by_run.empty else 0.0)
    dom_div = (evid['domain'].nunique() if not evid.empty and 'domain' in evid.columns else 0)

    lbl_counts = norm.get('sentiment_label', pd.Series([])).value_counts()
    pos_share = (lbl_counts.get('positive',0) / max(lbl_counts.sum(),1))

    col1.metric('Inclusion Rate', f'{inc*100:.1f}%')
    col2.metric('Sentiment Ø', f'{sent:+.2f}')
    col3.metric('Freshness Index', f'{fresh:.2f}')
    col4.metric('% Runs mit Belegen',
