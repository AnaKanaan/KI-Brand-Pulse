import os, time, pandas as pd, streamlit as st
from ki_rep_monitor import run_pipeline


st.set_page_config(page_title='KI-Reputation Monitor', layout='wide')
st.title('🔎 KI-Reputation Monitor — Final3')

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

    uploaded = st.file_uploader('Question Library (xlsx, optional)', type=['xlsx'])
    if uploaded is not None:
        question_xlsx = uploaded
    else:
        question_xlsx = 'ki_question_library.xlsx'

    run_btn = st.button('🚀 Run')

def parse_ids(s):
    if not s or not s.strip(): return None
    out=[]
    for p in s.split(','):
        p=p.strip()
        if not p: continue
        try: out.append(int(p))
        except: pass
    return out or None

if run_btn:
    if not os.getenv('OPENAI_API_KEY'):
        st.error('Bitte zuerst OpenAI API Key setzen.')
        st.stop()
    out_name = f'out_{int(time.time())}.xlsx'
    with st.spinner('Läuft …'):
        if isinstance(question_xlsx, str):
            q_path = question_xlsx
        else:
            q_path = f'/tmp/_qlib_{int(time.time())}.xlsx'
            with open(q_path, 'wb') as f: f.write(question_xlsx.getbuffer())

        st.sidebar.write("Questions columns:", list(pd.read_excel(q_path, sheet_name="Questions").columns))

        res = run_pipeline(
            brand=brand, topic=topic, market=market,
            languages=languages, profiles=profiles,
            question_xlsx=q_path, out_xlsx=out_name,
            domain_seed_csv='domain_type_seed.csv',
            coder_prompts_json='coder_prompts_passB.json',
            topn=int(topn), num_runs=int(num_runs),
            categories=categories, question_ids=parse_ids(question_ids_raw),
            comp1=comp1, comp2=comp2, comp3=comp3,
            temperature_chat_no=float(temp_no), temperature_chat_search=float(temp_auto),
            max_tokens=int(max_tokens), wrapper_mode=wrapper_mode
        )

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

        st.subheader('Runs'); st.dataframe(runs, use_container_width=True, hide_index=True)
        st.subheader('Evidence'); st.dataframe(evid, use_container_width=True, hide_index=True)
        st.subheader('Normalized (flattened JSON)'); st.dataframe(norm, use_container_width=True, hide_index=True)
        st.subheader('Config'); st.table(cfg)
else:
    st.info('Keys setzen (optional), konfigurieren und **Run** starten.')
