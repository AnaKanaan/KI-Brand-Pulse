import os
import io
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from ki_rep_monitor import run_pipeline

st.set_page_config(page_title="KI-Reputation Monitor", layout="wide")

st.title("🔎 KI-Reputation Monitor — Streamlit")
st.caption("Repliziert ChatGPT (ohne/mit Auto-Suche) & Google-Overview (CSE) • Normalisiert & reichert Evidenzen an • Export als Excel")

with st.sidebar:
    st.header("⚙️ Einstellungen")
    brand = st.text_input("Brand", "DAK")
    topic = st.text_input("Topic", "KI im Gesundheitswesen")
    market = st.text_input("Market (z. B. DE, CH, AT)", "DE")
    profiles = st.multiselect("Profiles", ["CHATGPT_NO_SEARCH","CHATGPT_SEARCH_AUTO","GOOGLE_OVERVIEW"],
                              default=["CHATGPT_NO_SEARCH","CHATGPT_SEARCH_AUTO","GOOGLE_OVERVIEW"])
    languages = st.multiselect("Languages", ["de","fr","it","rm","en"], default=["de"])
    topn = st.number_input("Google CSE Top-N", min_value=1, max_value=10, value=5, step=1)
    num_runs = st.number_input("Replikate je Frage", min_value=1, max_value=10, value=3, step=1)

    st.subheader("📄 Fragebibliothek")
    uploaded = st.file_uploader("Excel (optional)", type=["xlsx"])
    if uploaded is not None:
        question_xlsx = uploaded
    else:
        question_xlsx = "ki_question_library.xlsx"

    st.subheader("🔐 API Keys (Secrets)")
    st.write("Auf Streamlit Cloud in **App → Settings → Secrets** setzen: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`.")
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY nicht gesetzt.")
    if "GOOGLE_OVERVIEW" in profiles and not os.getenv("GOOGLE_API_KEY"):
        st.warning("GOOGLE_API_KEY/GOOGLE_CSE_ID nicht gesetzt (erforderlich für GOOGLE_OVERVIEW).")

    run_btn = st.button("🚀 Run")

placeholder = st.empty()

if run_btn:
    with st.spinner("Läuft… (Calls an OpenAI/Google – kann je nach Replikaten dauern)"):
        out_name = f"out_{int(time.time())}.xlsx"
        try:
            # Save uploaded file if present
            if isinstance(question_xlsx, str):
                q_path = question_xlsx
            else:
                q_path = f"/tmp/_questions_{int(time.time())}.xlsx"
                with open(q_path, "wb") as f:
                    f.write(question_xlsx.getbuffer())
            # Execute pipeline
            from ki_rep_monitor import run_pipeline as _run
            _run(brand=brand, topic=topic, market=market, languages=languages, profiles=profiles,
                 question_xlsx=q_path, out_xlsx=out_name, domain_seed_csv="domain_type_seed.csv",
                 coder_prompts_json="coder_prompts_passB.json", topn=int(topn), num_runs=int(num_runs))

            # Load and show
            xls = pd.ExcelFile(out_name)
            runs = pd.read_excel(xls, "Runs")
            norm = pd.read_excel(xls, "Normalized")
            evid = pd.read_excel(xls, "Evidence")

            st.success(f"Fertig: {out_name}")
            st.download_button("📥 Download Excel", data=open(out_name,"rb").read(), file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Runs")
                st.dataframe(runs, use_container_width=True)
            with c2:
                st.subheader("Evidence")
                st.dataframe(evid, use_container_width=True)

            st.subheader("Normalized (Flattened JSON)")
            st.dataframe(norm, use_container_width=True)

        except Exception as e:
            st.error(f"Fehler: {e}")
            st.stop()
else:
    st.info("Konfigurieren → **Run** starten.")
