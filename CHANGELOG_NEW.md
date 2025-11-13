# KI‑Reputation Monitor – Änderungsliste und Dokumentation

Dieses Dokument beschreibt die wichtigsten Änderungen und Erweiterungen gegenüber der vorherigen Version der **KI‑Reputation Monitor**‑App.  Der Fokus liegt auf funktionaler Verbesserung, konzeptioneller Passung und Nachvollziehbarkeit.  Aspekte wie Datenschutz und Sicherheit bleiben unverändert, da sie für diese Iteration nicht im Mittelpunkt standen.

## Übersicht der wichtigsten Neuerungen

| Bereich | Alt | Neu |
|-------|----|----|
| **Profile (Pass A)** | Drei Profile: `CHATGPT_NO_SEARCH`, `CHATGPT_SEARCH_AUTO` (mit Bing‑Suche), `GOOGLE_OVERVIEW` (Google CSE) | Fünf Profile: bestehende + zwei neue Gemini‑Profile: `GEMINI_NO_SEARCH` und `GEMINI_SEARCH_AUTO`.  Gemini verwendet das kostenlose Modell **gemini‑2.5‑flash**, wie in öffentlichen Vergleichen angegeben【416161300090141†L360-L365】.  Das freie Gemini‑Chat nutzt Google‑Suche für Echtzeitdaten【854845349309103†L431-L435】, weshalb `GEMINI_SEARCH_AUTO` die CSE‑API nutzt und anschließend via Gemini zusammenfasst. |
| **Pflicht‑Keys** | Gemini API Key optional; wenn nicht gesetzt, wurde `GOOGLE_OVERVIEW` über OpenAI zusammengefasst | **Gemini API Key ist Pflicht**: ohne gültigen Schlüssel bricht das Programm mit RuntimeError ab.  Dies stellt sicher, dass Gemini‑Anfragen immer authentifiziert sind. |
| **Fragenbibliothek** | Eine einzige Tabelle `Questions` mit Spalten `profile` und `num_runs` (bei jedem Eintrag hinterlegt). | Die Excel‑Bibliothek wurde komplett restrukturiert.  Jeder Sprache (`de`, `fr`, `it`, `rm`, `en`) ist ein eigenes Blatt gewidmet.  Spalten sind `question_id`, `question_text`, `language`, `category`, `intent`, `variant`.  Die Spalten `profile` und `num_runs` entfallen – die Zahl der Wiederholungen wird global via UI gesteuert.  Zusätzlich gibt es ein Blatt `DomainTaxonomy`, das aus der CSV `domain_type_seed.csv` generiert wird. |
| **Stakeholder‑Perspektive** | Nicht berücksichtigt.  Fragen bezogen sich immer auf eine generische Perspektive. | Das System unterstützt nun eine Stakeholder‑Liste.  Über das neue Eingabefeld in der Sidebar können Anwender auswählen, aus welcher Sicht die Frage gestellt wird (z. B. `Bewerber`, `Investor`, `Mitarbeitender`, `Business‑Kunde`, `Politischer Entscheider` usw.).  Ist in der Frage der Platzhalter `<STAKEHOLDER>` enthalten, wird er ersetzt; ansonsten wird bei deutschen Fragen automatisch ein Vorspann *„Aus Sicht eines {Stakeholder}: …“* vorangestellt.  Der Stakeholder fließt als Feld `stakeholder` in alle Ergebnis‑Daten (Runs, Normalized, Evidence) ein. |
| **Nummer der Wiederholungen** | Die Anzahl der Replikationen pro Frage war in der Excel‑Bibliothek (`num_runs`) hinterlegt und damit pro Frage unterschiedlich. | `num_runs` ist jetzt ein globaler Parameter.  Die UI bietet ein Feld „Replicates per question“, das für alle Fragen gilt.  Die Bibliothek enthält keine pro‑Frage‑Angabe mehr. |
| **ChatGPT Search Auto** | Nutzt das OpenAI Tooling für Websuche über Bing; komplizierte Tool‑Response‑Parsing. | Die Auto‑Suche verhält sich jetzt wie das ChatGPT‑Free‑Modell ohne Tools: es wird nur das Chat‑Modell angesprochen.  Anschließend werden Domains und URLs aus der Antwort mithilfe regulärer Ausdrücke extrahiert, sodass trotzdem Belege gesammelt werden. |
| **Gemini‑Integration** | Nur als Kurzfassung für `GOOGLE_OVERVIEW` optional genutzt. | Voll integrierte Gemini‑Profile.  `GEMINI_NO_SEARCH` ruft das kostenlose Modell `gemini‑2.5‑flash` direkt via API auf.  `GEMINI_SEARCH_AUTO` führt zunächst eine Google‑Custom‑Search durch und übergibt die Treffer gebündelt an Gemini, das daraus eine Antwort generiert.  Die Google‑Suche wird nur zum Sammeln von Fakten genutzt, die Antwort selbst kommt von Gemini. |
| **Evidence & Datumsextraktion** | Evidence bestand aus Domain‑Zitaten, angereichert mit Domain‑Typ und Freshness.  Das Veröffentlichungsdatum wurde nicht aus den Links ermittelt. | Für jede Evidence ohne `published_at` wird die verlinkte Seite abgerufen und nach gängigen Meta‑Tags (`datePublished`, `article:published_time` etc.) bzw. Datums‑Mustern durchsucht.  Dadurch kann das Alter der Quelle präziser berechnet werden und fließt in `age_days`, `freshness_bucket` und `freshness_index` ein. |
| **Token‑Limits (Pass A)** | Default `max_output_tokens` für Antworten lag bei 900 (bzw. 1600 bei Suche). | Die Standardwerte für `max_output_tokens` wurden auf **4000 Tokens** erhöht, um längere Antworten zu ermöglichen.  Das kann in der UI konfiguriert werden. |
| **UI/Streamlit** | Profiles beschränkten sich auf drei Optionen; kein Stakeholder‑Feld.  Debugging nur für Basic‑Events. | Erweiterte Sidebar mit Auswahl der neuen Gemini‑Profile und Stakeholder.  Default‑Token‑Limits angepasst.  Der Worker erhält jetzt die Stakeholder‑Liste und erzeugt für jede Kombination von Frage, Profil und Stakeholder einen eigenen Run.  Das Debug‑Panel listet zusätzlich Stakeholder und extrahierte Domain‑Belege auf. |

## Detaillierte Implementierungsänderungen

### 1. `ki_rep_monitor.py`

* **Neue Konstanten:** `DEFAULT_GEMINI_MODEL` setzt standardmäßig auf `gemini‑2.5‑flash` – das Modell, das laut öffentlichen Quellen im freien Gemini‑Chat verwendet wird【416161300090141†L360-L365】.  Kann per ENV Variable überschrieben werden.
* **Pflicht‑Keys:** Die Funktion `gemini_generate_text` fordert nun zwingend einen `GEMINI_API_KEY`.  Fehlt der Schlüssel, wird ein RuntimeError geworfen.
* **Domain‑Extraction:** Die Hilfsfunktion `extract_domains_from_text` extrahiert Domains aus Plain‑Text‑Antworten.  Sie sucht nach Mustern wie `(example.com)` oder nach nackten Domain‑Namen und erzeugt Evidence‑Einträge mit Snippets und Zeitpunkt der Entdeckung.
* **Datums‑Extraktion:** `extract_publication_date` holt HTML‑Seiten ab und durchsucht sie nach Meta‑Tags (`datePublished`, `article:published_time`, `og:published_time`) oder nach generischen Datumsmustern.  Das gefundene Datum wird als ISO‑Zeitstempel in UTC zurückgegeben.  Ist keines vorhanden, bleibt das Feld leer.
* **Enrichment:** In `enrich_evidence` wird für Evidence ohne `published_at` das Datum via `extract_publication_date` ergänzt.  Die Freshness‑Indizes werden entsprechend berechnet.
* **Neue LLM‑Wrapper:** Funktionen `call_gemini_no_search` und `call_gemini_search_auto` implementieren die Gemini‑Profile.  Die Search‑Variante nutzt `cse_list` (Google Custom Search) gefolgt von einer Geminí‑Zusammenfassung.  Alle Belege stammen aus den CSE‑Treffern, nicht aus der Modellantwort.
* **ChatGPT Search Auto:** `call_chat_search_auto` ruft nur noch das Chat‑Modell ohne Tool‑Zugriff auf und extrahiert anschließend Domains aus dem Antworttext.  Dies entspricht der „Bing‑Suche“ im freien ChatGPT, wo ebenfalls keine Quellen angezeigt werden, aber indirekte Hinweise im Text vorkommen können.
* **Stakeholder‑Loop:** `run_pipeline` akzeptiert jetzt eine Liste von Stakeholdern.  Für jede Frage wird die Kombination aus Profil, Stakeholder und Replikat durchlaufen.  Der Stakeholder wird sowohl im Prompt (Ersetzung des Platzhalters oder Voranstellung) als auch in den Ergebnisdaten (`stakeholder` in Normalized) geführt.  Der Run‑Identifier (`run_id`) enthält den Stakeholder.
* **Global `num_runs`:** Die Anzahl der Replikationen wird nicht mehr aus der Bibliothek gelesen, sondern über den Parameter `num_runs` gesetzt.  Die Bibliothek enthält keine `num_runs`‑Spalte mehr.
* **Tokengrenzen:** Der Parameter `max_tokens` wird von der UI mit bis zu 4 000 Tokens befüllt.  Für Suchprofile (`*_SEARCH_AUTO`) kann `passA_search_tokens` genutzt werden, um einen höheren Grenzwert (Standard ebenfalls 4 000) zu übergeben.

### 2. `coder_prompts_passB.json`

* Das Datei‑Mapping wurde neu erstellt und enthält nun die neuen Profile `GEMINI_NO_SEARCH` und `GEMINI_SEARCH_AUTO`.
* Das Zielschema weist zusätzlich das Feld `stakeholder` vom Typ `string` aus.  Dieses Feld wird vom Code nach Pass B eingefügt und enthält die Stakeholder‑Perspektive, aus der die Frage gestellt wurde.
* Die Dateien sind für jede Sprache separat definiert; das JSON‑Schema ist identisch, lediglich die Beschreibung des Tasks („Normalisiere die Rohantwort …“) ist sprachabhängig.

### 3. Fragenbibliothek (`ki_question_library.xlsx`)

* Die Bibliothek wurde aus der ursprünglichen Datei rekonstruiert.  Es gibt nun pro Sprache (`de`, `fr`, `it`, `rm`, `en`) ein eigenes Blatt.  Jedes Blatt enthält die Spalten `question_id`, `question_text`, `language`, `category`, `intent` und `variant`.  Spalten wie `profile`, `brand`, `topic`, `market`, `competitors` und `num_runs` wurden entfernt, da diese Werte zur Laufzeit vom Benutzer eingegeben werden.
* Ein zusätzliches Blatt `DomainTaxonomy` enthält die Domain‑Typen aus der Datei `domain_type_seed.csv`.  Dieses Blatt dient lediglich der Transparenz; die eigentliche Zuordnung erfolgt programmatisch über die CSV.

### 4. Streamlit‑UI (`streamlit_app.py`)

* **Profile‑Auswahl:** Die Sidebar listet nun auch `GEMINI_NO_SEARCH` und `GEMINI_SEARCH_AUTO` als wählbare Profile.
* **Stakeholder‑Auswahl:** Unterhalb der Sprachwahl gibt es eine Mehrfachauswahl „Stakeholders“.  Voreinstellung ist `generic`.  Bei leerer Auswahl wird automatisch `generic` verwendet.
* **Tokens:** Die Standardwerte für `max_output_tokens` (ohne Suche und mit Suche) wurden auf 4 000 erhöht.
* **Worker‑Aufruf:** Der Worker übergibt jetzt die Stakeholder‑Liste an `run_pipeline`.  Dadurch wird für jedes Stakeholder‑Profil eine eigene Ausführung gestartet.
* **Debug‑Panel:** Das Live‑Protokoll zeigt nun auch die Stakeholder und extrahierte Evidence‑Quellen an.  Dadurch können Antworten, Belege und die Zuordnung zu Stakeholder‑Perspektiven transparent nachvollzogen werden.

### 5. Weitere Anpassungen

* **Dokumentation:** Dieses Changelog dient als Ablöse‑Dokumentation.  Die ursprüngliche README wurde nicht verändert, um Abwärtskompatibilität zu gewährleisten.  Anwender finden hier jedoch alle Neuerungen im Detail.
* **Dependency‑Lock:** Es wurden keine zusätzlichen Bibliotheken eingeführt.  Die vorhandenen Abhängigkeiten (`pandas`, `openpyxl`, `requests`, `tldextract`, `google‑genai`, `streamlit`) bleiben bestehen.

## Nutzungshinweise

1. **API‑Schlüssel setzen:** Im UI‑Expander „🔐 API‑Keys“ müssen OpenAI‑, Google‑ und Gemini‑Schlüssel angegeben werden.  Insbesondere der Gemini‑Schlüssel ist nun obligatorisch.
2. **Fragen konfigurieren:** Über die Sidebar können Markenname, Thema, Markt, Wettbewerber, Profile, Sprachen, Stakeholder, Kategorien und die Anzahl der Wiederholungen eingestellt werden.  Es empfiehlt sich, für Stakeholder mehrere Perspektiven zu wählen, um Unterschiede in der Wahrnehmung zu analysieren.
3. **Lauf starten und überwachen:** Nach Klick auf **Run** startet der Worker‑Thread.  Fortschritt, ETA und Debug‑Informationen werden live angezeigt.  Die Ausführung kann jederzeit abgebrochen werden.
4. **Ergebnisse herunterladen:** Nach Abschluss steht eine Excel‑Datei mit den Sheets `Runs`, `Normalized`, `Evidence`, `Config` und `RawAnswers` zur Verfügung.  Diese kann direkt in weitere Analyse‑Tools importiert werden.

## Quellenangaben

* Öffentliche Berichte bestätigen, dass das kostenlose Gemini‑Chat auf dem Modell **Gemini 2.5 Flash** basiert【416161300090141†L360-L365】.  Das Pro‑Modell (Gemini 2.5 Pro) ist nur limitiert zugänglich.  Beide Gemini‑Varianten nutzen Google‑Suche für Echtzeit‑Daten【854845349309103†L431-L435】.

* Die Entscheidung, das ChatGPT‑Search‑Profil ohne Tools zu realisieren, spiegelt die Einschränkungen der freien ChatGPT‑Version wider – sie unterstützt keine echten Web‑Tools, aber domänenspezifische Hinweise können im Text vorkommen.
