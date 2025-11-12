# streamlit_app.py
# Streamlit-UI mit Event-Queue (keine st.*-Aufrufe im Worker), Start/Abbrechen, ETA/Watchdog,
# Debug-Panel, Key-Handling (OpenAI/Google/Gemini nur in Session), Modell-Defaults wie gefordert.

import os, time, json, threading
from queue import Queue, Empty
import pandas as pd
import streamlit as st
from ki_rep_monitor import run_pipeline

st.set_page_config(page_title='KI-Reputation Monitor', layout='wide')
st.title('🔎 KI-Reputation Monitor — Final3')

# --- Globaler Debug-Schalter (während Run gesperrt) ---
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False

is_running = bool(st.session_state.runner.get("thread") and st.session_state.runner["thread"].is_alive())

st.checkbox("Debug-Modus (Events unten anzeigen)", key="debug_mode", disabled=is_running)
if st.session_state.debug_mode:
    st.checkbox("Raw API-Payloads (redigiert)", key="show_raw", disabled=is_running)

# ---- Session-State ----
if "runner" not in st.session_state:
    st.session_state.runner = {"thread": None, "cancel": None, "start": None, "run_id": 0}
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []
if "last_event_ts" not in st.session_state:
    st.session_state.last_event_ts = 0.0

# Clean up toter Thread
rt = st.session_state.runner["thread"]
if rt and not rt.is_alive():
    st.session_state.runner.update({"thread": None, "cancel": None, "start": None})

# ---------------- API Keys (nur Session) ----------------
with st.expander('🔐 API-Keys (nur Session, keine Speicherung)'):
    openai_key = st.text_input('OpenAI API Key', type='password', placeholder='sk-...')
    google_key = st.text_input('Google API Key (CSE)', type='password', placeholder='AIza...')
    google_cx  = st.text_input('Google CSE ID (cx)', type='password', placeholder='custom search engine id')
    gemini_key = st.text_input('Gemini API Key', type='password', placeholder='AIza... oder ***')

    if st.button('Apply Keys'):
        if openai_key: os.environ['OPENAI_API_KEY'] = openai_key
        if google_key: os.environ['GOOGLE_API_KEY'] = google_key
        if google_cx:  os.environ['GOOGLE_CSE_ID']  = google_cx
        if gemini_key: os.environ['GEMINI_API_KEY'] = gemini_key
        st.success('Keys gesetzt (nur Session).')

# ---------------- Sidebar Controls ----------------
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
    max_tokens = st.number_input('max_output_tokens', 100, 4000, 900, 50)
    wrapper_mode = st.selectbox('Pass-A Wrapper', ['free_emulation','stabilized'], index=0)

    st.markdown("---")
    st.markdown("**Model Settings**")
    # Vorgabe exakt wie gewünscht:
    model_pass_a = st.text_input("Model (Pass A: Antwort)", os.getenv("MODEL_PASS_A", "gpt-5-chat-latest"))
    model_pass_b = st.text_input("Model (Pass B: Codierung)", os.getenv("MODEL_PASS_B", "gpt-5"))
    use_gemini_overview = st.checkbox("Gemini für Overview verwenden (falls GEMINI_API_KEY gesetzt)", value=True)

    uploaded = st.file_uploader('Question Library (xlsx, optional)', type=['xlsx'])
    question_xlsx = uploaded if uploaded is not None else 'ki_question_library.xlsx'

    # Start/Abbrechen
    run_btn = st.button('🚀 Run')
    abort_btn = st.button('⛔ Abbrechen')

def parse_ids(s: str):
    if not s or not s.strip(): return None
    out = []
    for p in s.split(','):
        p = p.strip()
        if not p: continue
        try: out.append(int(p))
        except: pass
    return out or None

# Abbrechen
if abort_btn and st.session_state.runner["cancel"] is not None:
    st.session_state.runner["cancel"].set()
    st.info("Abbruch angefordert – bitte kurz warten …")

# ---------------- Run-Handler ----------------
if run_btn:
    if not os.getenv('OPENAI_API_KEY'):
        st.error('Bitte zuerst OpenAI API Key setzen.')
        st.stop()

    # Modell-Overrides → Env (werden in ki_rep_monitor.py gelesen)
    os.environ["MODEL_PASS_A"] = model_pass_a or "gpt-5-chat-latest"
    os.environ["MODEL_PASS_B"] = model_pass_b or "gpt-5"

    # Vorherigen Run ggf. abbrechen
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
        with open(q_path, 'wb') as f:
            f.write(question_xlsx.getbuffer())

    # Quick-Preview
    try:
        qdf = pd.read_excel(q_path, sheet_name="Questions")
        # Pre-Run-Zusammenfassung: Wie viele Fragen bleiben nach Filtern?
        try:
            qdf_norm = qdf.copy()
            qdf_norm.columns = [str(c).strip() for c in qdf_norm.columns]
            lower = {c.lower(): c for c in qdf_norm.columns}
            # tolerantes Mapping wie im Backend
            if "id" in lower:    qdf_norm = qdf_norm.rename(columns={ lower["id"]: "question_id" })
            if "query" in lower: qdf_norm = qdf_norm.rename(columns={ lower["query"]: "question_text" })

            qdf_norm["language"] = qdf_norm["language"].astype(str).str.strip().str.lower()
            qdf_norm["category"] = qdf_norm["category"].astype(str).str.strip()

            orig = len(qdf_norm)
            langs = [str(l).strip().lower() for l in languages if str(l).strip()] if languages else None
            cats  = {str(c).strip().upper() for c in categories if str(c).strip()} if categories else None

            if langs:
                qdf_norm = qdf_norm[qdf_norm["language"].isin(langs)].copy()
            if cats:
                qdf_norm = qdf_norm[qdf_norm["category"].str.upper().isin(cats)].copy()

            if question_ids_raw:
                ids = [int(x.strip()) for x in question_ids_raw.split(",") if x.strip().isdigit()]
            if ids:
                qdf_norm = qdf_norm[qdf_norm["question_id"].isin(ids)].copy()

            st.sidebar.info(f"Fragen nach Filtern: {len(qdf_norm)} / {orig}")
            st.sidebar.caption(f"Sprachen im Sheet: {sorted(list(set(qdf['language'].astype(str)) ))[:10]}")
            st.sidebar.caption(f"Kategorien im Sheet: {sorted(list(set(qdf['category'].astype(str)) ))[:10]}")
        except Exception as e:
            st.sidebar.warning(f"Pre-Run-Filter-Vorschau nicht möglich: {e}")
            st.sidebar.write("Questions sheet columns:", list(qdf.columns))
            st.sidebar.write("Preview:", qdf.head(3))
    except Exception as e:
        st.sidebar.error(f"Kann 'Questions' nicht lesen: {e}")
        st.stop()

    # UI-Platzhalter
    status = st.status("Pipeline startet …", state="running")
    prog   = st.progress(0)
    eta_box = st.empty()
    step_box = st.empty()
    health_box = st.empty()
    stall_warning_box = st.empty()
    debug_mode = bool(st.session_state.debug_mode)
    show_raw   = bool(st.session_state.show_raw) if debug_mode else False
    debug_expander = st.expander("🪵 Debug-Protokoll", expanded=False) if debug_mode else None
    debug_area = debug_expander.container() if debug_mode else None

    # Event-Queue
    event_q = Queue()
    last_done = {"done": 0, "total": 0}
    stall_limit = 20  # Sek. ohne Events → Warnung

    def _fmt_eta(sec: int) -> str:
        m, s = divmod(max(int(sec), 0), 60)
        return f"{m:02d}:{s:02d}"

    def _process_event_in_main_thread(ev: dict):
        st.session_state.debug_log.append(ev)
        st.session_state.last_event_ts = time.time()
        meta = ev.get("meta", {})

        pct = meta.get("pct")
        if pct is None and meta.get("done") and meta.get("total"):
            try:
                pct = int(5 + 95 * (meta["done"] / max(meta["total"], 1)))
            except Exception:
                pct = None
        if pct is not None:
            try: prog.progress(int(pct))
            except Exception: pass

        if "eta_s" in meta:
            eta_box.markdown(f"**ETA:** {_fmt_eta(meta['eta_s'])} min")

        step_box.write(f"**{ev.get('phase','')}** — {ev.get('msg','')}")

        if "done" in meta and "total" in meta:
            last_done["done"] = int(meta["done"]); last_done["total"] = int(meta["total"])

        if debug_mode and debug_area is not None:
            last = st.session_state.debug_log[-12:]
            lines = []
            for e in last:
                ts = time.strftime('%H:%M:%S', time.localtime(e["t"]))
                phase = e.get("phase",""); msg = e.get("msg","")
                lines.append(f"- {ts} **{phase}** — {msg}")
                m = e.get("meta", {})
                if show_raw and "raw" in m:
                    lines.append(f"    ```\n{m['raw']}\n    ```")
                elif "prompt_excerpt" in m:
                    lines.append(f"    Prompt: `{m['prompt_excerpt']}`")
                elif "parsed_preview" in m:
                    lines.append(f"    Parsed: `{m['parsed_preview']}`")
            debug_area.markdown("\n".join(lines) if lines else "_(noch keine Events)_")

    def progress_hook(ev: dict):
        try:
            event_q.put_nowait(ev)
        except Exception:
            pass

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
                max_tokens=int(max_tokens), wrapper_mode=wrapper_mode,
                progress=progress_hook, debug=debug_mode, cancel=cancel_event,
                use_gemini_overview=bool(use_gemini_overview)
            )
        except Exception as e:
            result["err"] = e

    t0 = time.time()
    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    st.session_state.runner.update({"thread": th, "cancel": cancel_event, "start": t0, "run_id": st.session_state.runner["run_id"] + 1})

    # Herzschlag + Event-Verarbeitung
    last_hb = 0; last_stall_notice = 0
    while th.is_alive():
        try:
            while True:
                ev = event_q.get_nowait()
                _process_event_in_main_thread(ev)
        except Empty:
            pass

        now = time.time()
        if now - last_hb >= 1:
            elapsed = int(now - t0); m, s = divmod(elapsed, 60)
            since = int(now - (st.session_state.last_event_ts or now))
            health_box.markdown(f"**Health:** letzte Event-Aktualisierung vor {since}s · Fortschritt {last_done['done']}/{max(last_done['total'],1)}")
            status.update(label=f"Läuft … ({m:02d}:{s:02d} elapsed)", state="running")
            last_hb = now

        if st.session_state.last_event_ts:
            silence = now - st.session_state.last_event_ts
            if silence > stall_limit and now - last_stall_notice > 5:
                stall_warning_box.warning(
                    f"Seit {int(silence)}s keine neuen Events. Prüfe Netzwerk/Keys. "
                    f"Du kannst den Lauf mit **⛔ Abbrechen** stoppen."
                )
                last_stall_notice = now

        time.sleep(0.1)

    # Rest-Events abholen
    try:
        while True:
            ev = event_q.get_nowait()
            _process_event_in_main_thread(ev)
    except Empty:
        pass

    # Abschluss
    prog.progress(100)
    if cancel_event.is_set() and not result["res"]:
        status.update(label="Abgebrochen", state="error")
        st.warning("Lauf wurde abgebrochen.")
        if st.session_state.debug_log:
            dbg_path = f"/tmp/debug_log_{int(time.time())}.json"
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.debug_log, f, ensure_ascii=False, indent=2)
            with open(dbg_path, "rb") as f:
                st.download_button("🧾 Debug-Log herunterladen (JSON)", f,
                                   file_name=os.path.basename(dbg_path), mime="application/json")
        st.stop()
    elif result["err"]:
        status.update(label="Fehlgeschlagen", state="error")
        st.exception(result["err"])
        if st.session_state.debug_log:
            dbg_path = f"/tmp/debug_log_{int(time.time())}.json"
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.debug_log, f, ensure_ascii=False, indent=2)
            with open(dbg_path, "rb") as f:
                st.download_button("🧾 Debug-Log herunterladen (JSON)", f,
                                   file_name=os.path.basename(dbg_path), mime="application/json")
        st.stop()
    else:
        status.update(label="Fertig", state="complete")

    # ---------------- Auswertung & Anzeige ----------------
    xls = pd.ExcelFile(out_name)
    runs = pd.read_excel(xls, 'Runs')
    norm = pd.read_excel(xls, 'Normalized')
    evid = pd.read_excel(xls, 'Evidence')
    cfg  = pd.read_excel(xls, 'Config')

    st.success(f'Fertig: {out_name}')
    with open(out_name, 'rb') as fbin:
        st.download_button('📥 Download Excel', data=fbin.read(), file_name=out_name)

    st.subheader('📊 KPIs')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if not norm.empty:
        inc = norm.get('inclusion', pd.Series([False]*len(norm))).fillna(False).astype(bool).mean()
        try:
            sent = norm.get('aspect_scores.sentiment', pd.Series([0.0]*len(norm))).astype(float).fillna(0).mean()
        except Exception:
            sent = 0.0
        fresh = norm.get('freshness_index', pd.Series([0.0]*len(norm))).astype(float).fillna(0).mean()
    else:
        inc = sent = fresh = 0.0

    if not evid.empty:
        evid_by_run = evid.groupby('run_id').size()
        ev_rate = float(evid_by_run.gt(0).mean()) if not evid_by_run.empty else 0.0
        dom_div = int(evid['domain'].nunique()) if 'domain' in evid.columns else 0
    else:
        ev_rate = 0.0; dom_div = 0

    lbl_counts = norm.get('sentiment_label', pd.Series(dtype=object)).value_counts() if not norm.empty else pd.Series(dtype=int)
    pos_share = (lbl_counts.get('positive', 0) / max(int(lbl_counts.sum()), 1)) if not lbl_counts.empty else 0.0

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
            fb = evid['freshness_bucket'].value_counts().reindex(
                ['today','≤7d','≤30d','≤90d','≤365d','>365d','unknown']
            ).fillna(0).reset_index()
            fb.columns = ['bucket','count']
            st.bar_chart(fb.set_index('bucket'))
        else:
            st.info('Keine Evidence-Daten.')

    st.markdown('### Profile × Language — Inclusion Rate')
    try:
        inc_pf = norm.assign(incl=norm['inclusion'].fillna(False).astype(bool)) \
                     .groupby(['profile','language'])['incl'].mean().reset_index()
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

    st.subheader('Runs'); st.dataframe(runs, use_container_width=True, hide_index=True)
    st.subheader('Evidence'); st.dataframe(evid, use_container_width=True, hide_index=True)
    st.subheader('Normalized (flattened JSON)'); st.dataframe(norm, use_container_width=True, hide_index=True)
    st.subheader('Config'); st.table(cfg)

    if st.session_state.debug_log:
        dbg_path = f"/tmp/debug_log_{int(time.time())}.json"
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.debug_log, f, ensure_ascii=False, indent=2)
        with open(dbg_path, "rb") as f:
            st.download_button("🧾 Debug-Log herunterladen (JSON)", f,
                               file_name=os.path.basename(dbg_path), mime="application/json")
else:
    st.info('Keys setzen (optional), konfigurieren und **Run** starten.')
