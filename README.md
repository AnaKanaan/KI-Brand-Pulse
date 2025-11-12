KI-Reputation Monitor – Streamlit (Final3)

Repliziert drei UX-Profile für eure KI-Reputationsmessung:

CHATGPT_NO_SEARCH – gpt-5-chat-latest ohne Tools (Responses API)

CHATGPT_SEARCH_AUTO – gpt-5-chat-latest mit tools: [{type: "web_search"}] und tool_choice: "auto"

GOOGLE_OVERVIEW – Google Custom Search JSON API (Top-N) → LLM-Übersicht nur aus Treffern
– Optional: Kurzfassung via Gemini (falls GEMINI_API_KEY gesetzt), sonst OpenAI

Pass B (Normalisierung): Strukturierte Kodierung der Rohantworten in ein striktes JSON-Schema (mehrsprachig), inkl. deterministischem Enrichment (Domain-Typ, Freshness).
Output als Excel: Runs, Normalized, Evidence, Config.

✨ Was ist neu (gegenüber deiner vorherigen Version)

Modellwahl präzisiert

Pass A: fix gpt-5-chat-latest für Chat-UX (kein Sampling).

Pass B: gpt-5 mit reasoning: {"effort":"medium"} und response_format: {"type":"json_object"}.

Parameter-Guard: Entfernt unzulässige Sampling-Parameter (temperature, top_p, logprobs, n) für GPT-5/Familie automatisch.

Optionale Gemini-Kurzfassung im Profil GOOGLE_OVERVIEW (falls GEMINI_API_KEY in der Session).

Transparenz & Kontrolle

Live-Fortschritt + ETA + Health/Watchdog (Stall-Erkennung).

Abbrechen-Button (sauberer Cancel).

Debug-Panel mit Event-Timeline (redigierte Payloads/Antworten) + Download des Debug-Logs (JSON).

Keine st.*-Aufrufe im Worker-Thread → kein ScriptRunContext-Spam mehr.

Robuste Questions-Validierung: Tolerantes Spalten-Mapping (z. B. id→question_id, query→question_text) und klare Fehlermeldungen.

Neue Dateien & Struktur: prompts/pass_a_wrappers.json, aktualisierte coder_prompts_passB.json, domain_type_seed.csv, optional .streamlit/config.toml.

Requirements aktualisiert (Python 3.13-kompatibel), inkl. google-genai.

🧱 Projektstruktur
.
├─ streamlit_app.py
├─ ki_rep_monitor.py
├─ coder_prompts_passB.json
├─ domain_type_seed.csv
├─ ki_question_library.xlsx
├─ prompts/
│  └─ pass_a_wrappers.json
├─ requirements.txt
└─ .streamlit/
   └─ config.toml         # optional, s. unten


ki_question_library.xlsx – Sheet „Questions“ (Pflichtspalten):

question_id (int)

question_text (string; Platzhalter ok: <BRAND>, <TOPIC>, <MARKET>, <COMP1>, <COMP2>, <COMP3>)

language („de“, „en“, „fr“, „it“, „rm“)

category (frei, z. B. „BRANDED“, „RISK“, „BENCHMARK“…)

intent (int)

variant (int)

📦 Requirements

requirements.txt

streamlit==1.39.0
pandas==2.3.3
openpyxl==3.1.5
requests==2.32.3
tldextract==5.1.2
google-genai==0.3.0


Getestet mit Python 3.13 (entspricht deinen Cloud-Logs).

🔑 Schlüssel (nur Session)

Schlüssel werden nur in der Session gesetzt (UI-Expander „🔐 API-Keys“) – keine Speicherung auf Disk:

OPENAI_API_KEY

GOOGLE_API_KEY + GOOGLE_CSE_ID (für Google CSE)

GEMINI_API_KEY (optional; nur für Gemini-Kurzfassung in GOOGLE_OVERVIEW)

Alternativ kannst du die ENV-Variablen auf deiner Plattform vordefinieren (dann ist die Eingabe im UI optional).

🚀 Lokal starten
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional per ENV (ansonsten im UI setzen):
# export OPENAI_API_KEY=sk-...
# export GOOGLE_API_KEY=AIza...
# export GOOGLE_CSE_ID=your_cx
# export GEMINI_API_KEY=...

streamlit run streamlit_app.py

☁️ Streamlit Community Cloud

Repo pushen.

New app → Hauptdatei: streamlit_app.py.

App öffnen → im UI-Expander 🔐 API-Keys setzen (pro Session).
(Optional: Environment Variables in den App-Settings hinterlegen, dann entfällt die manuelle Eingabe.)

UI konfigurieren → Run.

🖱️ Bedienung & Optionen

Profiles: CHATGPT_NO_SEARCH, CHATGPT_SEARCH_AUTO, GOOGLE_OVERVIEW (mehrfach wählbar).

Model Settings (Sidebar):

Pass A (Antwort): Default gpt-5-chat-latest (überschreibbar).

Pass B (Codierung): Default gpt-5 (überschreibbar) – reasoning=medium ist fest im Code hinterlegt.

Gemini für Overview: Checkbox aktivieren (nur wirksam, wenn GEMINI_API_KEY gesetzt).

Wrappers: free_emulation (roh) oder stabilized (leichter Rahmen).

Laufsteuerung: Fortschrittsbalken, ETA, Health-Anzeige, ⛔ Abbrechen.

Debug: Debug-Modus + Anzeige roher (redigierter) Requests/Responses, Download des Event-Logs (JSON).

🧠 Architektur (Kurz)

Pass A: gpt-5-chat-latest; für Auto-Suche: tools: [{type:"web_search"}], tool_choice:"auto".

GOOGLE_OVERVIEW: Google CSE (Top-N) → Kurzfassung via Gemini (wenn Key) sonst via OpenAI.

Pass B: gpt-5 mit reasoning={"effort":"medium"} + response_format={"type":"json_object"}.

Parameter-Guard: Sampling-Parameter werden bei GPT-5-Familien automatisch entfernt.

Threading: Worker erzeugt Events → UI rendert (keine st.* im Worker).

📤 Output

Excel mit 4 Sheets:

Runs – Metadaten je Run (Profil, Sprache, Zeit, Provider/Modell)

Normalized – flach normalisierte JSON-Antworten aus Pass B (inkl. Scores/Labels)

Evidence – Quellen inkl. Domain-Typ, Freshness-Bucket

Config – Laufkonfiguration (Wrapper-Mode, Profile)

🧪 Verifikation

Im Debug-Panel siehst du pro Call:

Pass A: api_call_1_request → model=gpt-5-chat-latest

Pass B: normalize_request → model=gpt-5, reasoning="medium"

Latenzen, redigierte Payloads/Antworten, Fortschritt/ETA

Debug-Log als JSON herunterladen → Audit/Fehlersuche offline.

🛠️ Troubleshooting

KeyError: 'question_id'
→ Im Sheet „Questions“ fehlen Pflichtspalten oder sind falsch benannt. Erlaubtes Mapping: id→question_id, query→question_text.
→ Prüfe außerdem, dass language, category, intent, variant vorhanden sind.

OpenAI 4xx/5xx
→ Key fehlt/falsch, Rate-Limit oder Payload ungültig. Sieh ins Debug-Panel (Event api_call_1_response / Fehlermeldung).

Google CSE 403/429
→ Quota/Abrechnung prüfen, GOOGLE_API_KEY + GOOGLE_CSE_ID korrekt? topn ggf. reduzieren.

Gemini-Fehler/keine Antwort
→ GEMINI_API_KEY nicht gesetzt oder Modell nicht erreichbar. Fallback (OpenAI-Kurzfassung) greift automatisch.

App „hängt“
→ Health-Anzeige zeigt „letzte Event-Aktualisierung …s“. Bei Stillstand ⛔ Abbrechen und Debug-Log herunterladen.

ScriptRunContext-Warnings
→ Sollten verschwunden sein (keine st.* im Worker). Falls sie auftauchen: Stelle sicher, dass du keine Streamlit-Calls in eigenen Threads machst.

🔒 Sicherheit & Datenschutz

Keys werden nur in der Session gesetzt (UI), nicht gespeichert.

Debug-Ausgaben redigieren automatisch Geheimnisse (Tokens).

Evidence/Antworten werden nur lokal in der erzeugten Excel gespeichert.

🗂️ Optional: .streamlit/config.toml
[server]
headless = true
runOnSave = true

[client]
showSidebarNavigation = true

[logger]
level = "info"

⬇️ Beispiel-CSV für domain_type_seed.csv
domain_type,example_domains,tld_hints,keyword_hints
news,nzz.ch;zeit.de,.ch;.de,zeitung;news;bericht
company,siemens.com;nestle.com,.com,investor relations;press release;pressemitteilung
social,twitter.com;linkedin.com,.com,tweet;linkedin;post
blog,medium.com;substack.com,.com,blog;newsletter;meinung
gov,admin.ch;.gv.at;.gov,.ch;.at;.gov,amtlich;behörde;verordnung;gesetz
other,,,


Stand: automatisch generiert nach Integration der Final3-Änderungen (Fortschritt/ETA/Abbruch, Debug-Events, Gemini-Option, Pass-B-Reasoning, Param-Guard, Dateien & Requirements).
