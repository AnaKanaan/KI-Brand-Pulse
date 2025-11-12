# KI-Reputation Monitor – Streamlit

Repliziert drei UX-Profile für eure KI-Reputationsmessung:

1. **CHATGPT_NO_SEARCH** – `gpt-5-chat-latest` *ohne* Tools (Responses API)  
2. **CHATGPT_SEARCH_AUTO** – `gpt-5-chat-latest` *mit* `web_search` und `tool_choice: "auto"`  
3. **GOOGLE_OVERVIEW** – Google Custom Search JSON API (Top-N) → LLM-Übersicht *nur* aus Treffern

**Pass B** normalisiert die Rohantworten in ein striktes JSON-Schema (5 Sprachen) und reichert deterministisch an (Domain-Typ, Freshness).  
**Output** als Excel: `Runs`, `Normalized`, `Evidence`.

---

## 🚀 Lokal starten

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
export GOOGLE_CSE_ID=your_cx

streamlit run streamlit_app.py
```

> Für `GOOGLE_OVERVIEW` ist `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` erforderlich.

---

## ☁️ Streamlit Community Cloud

1. Dieses Repo auf GitHub pushen (z. B. `ki-rep-monitor`).
2. Auf [share.streamlit.io](https://share.streamlit.io) **New app** → Repo + Branch wählen → Hauptdatei: `streamlit_app.py`.
3. Unter **App → Settings → Secrets** setzen:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GOOGLE_API_KEY = "AIza..."
   GOOGLE_CSE_ID  = "your_cx"
   OPENAI_BASE_URL = "https://api.openai.com/v1"
   ```
4. Deploy. Danach UI konfigurieren und **Run** drücken.

---

## 📁 Struktur

```
.
├─ streamlit_app.py
├─ ki_rep_monitor.py
├─ coder_prompts_passB.json
├─ domain_type_seed.csv
├─ ki_question_library.xlsx
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml   # Template (Secrets kommen in Streamlit Cloud)
```

---

## 🔎 Hinweise

- **ChatGPT ohne Suche** mappt auf Responses API ohne Tools.  
- **Auto-Suche** nutzt Websuche via `tools: [{type: "web_search"}]` und `tool_choice: "auto"`.  
- **AI Overview** hat keinen offiziellen API-Endpunkt; die Lösung nutzt **Google CSE** und zwingt die Übersicht, nur aus Treffern zu schreiben.

Stand: 2025-11-11T16:59:31.669607Z
