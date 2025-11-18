# KI‑Brand‑Pulse — Konsolidierter Stand (18.11.2025)

> **Claims**
>
> **„Was erzählt die KI‑Landschaft aktuell über meine Marke / mein Thema – und auf welcher Wissensbasis?“**  
> *„Wir messen, wie KI‑Assistenten deinen Ruf formen: welche Antworten Nutzer*innen bekommen, welche Quellen dahinterstehen und wie stabil diese Bilder sind.“*

Dieses Readme beschreibt den aktuellen, konsolidierten Stand der App nach allen Anpassungen der letzten Runden.

---

## Inhalt
- [Ziel & Überblick](#ziel--überblick)
- [Kernfunktionen](#kernfunktionen)
- [Profile (NO_SEARCH, SEARCH_AUTO, GOOGLE_OVERVIEW)](#profile-no_search-search_auto-google_overview)
- [KPIs & Definitionen](#kpis--definitionen)
- [Answer Quality (quality_flags) & Scores](#answer-quality-quality_flags--scores)
- [Stabilitätsmetriken (Multi-Runs)](#stabilitätsmetriken-multi-runs)
- [Evidence-Enrichment & Domain-Typisierung](#evidence-enrichment--domain-typisierung)
- [Stakeholder‑Bibliothek & Prompt-Prefixing](#stakeholder-bibliothek--prompt-prefixing)
- [Fragebibliothek (Excel) & Intents](#fragebibliothek-excel--intents)
- [Konfiguration & Laufzeit](#konfiguration--laufzeit)
- [Ausgaben / Exporte](#ausgaben--exporte)
- [Changelog (Auszug)](#changelog-auszug)
- [Grenzen & Hinweise](#grenzen--hinweise)

---

## Ziel & Überblick

**KI‑Brand‑Pulse** benchmarkt Antworten populärer LLM‑Assistenten zu einer Marke/einem Thema. Die App erfasst:
- **Was** geantwortet wird (Narrative, Sentiment, Visibility, Inclusion),
- **Worauf** sich die Antwort stützt (Quellen/Evidence, Domain‑Typen, Aktualität),
- **Wie stabil** Antworten über **mehrere Runs** sind (Agreement/Overlap),
- **Wie qualitativ** die Antworten sind (Answer‑Quality‑Flags + Score).

Das System beruht auf **zwei Pässen**:
- **Pass A**: Antwort erzeugen (ohne/mit Suche — je nach Profil)
- **Pass B**: **Sachlich codieren** (Visibility, Sentiment, Narrative, Risks, **Answer‑Quality**, …) gemäß strengen Prompts und Rückgabe‑Schema.

---

## Kernfunktionen

- **Feste Modelle & Tokenlimits**: Modelle und Output‑Längen sind **im Code fix**; **keine** UI‑Einstellung.
- **Kein Temperature/Top‑p**: Für GPT‑5/5.1 (Responses API) **entfernt**; **nicht** unterstützt.
- **Gemini‑Wrapper**: 
  - *No‑Search*: direkter Gemini‑Call.
  - *Search‑Auto*: Gemini mit `google_search`‑Tool, inkl. extrahierter Zitate.
- **Evidence‑Enrichment**: Normalisierung, Freshness, Domain‑Typisierung + **Original‑Objekt** (`original`). 
- **Stakeholder‑Prefixing**: Sprachspezifische Präfixe über **Excel‑Bibliothek**.
- **Intents**: Konsistente Intent‑Bedeutungen in **fünf Sprachen**.
- **Multi‑Runs** + **Stabilität**: num_runs pro Frage/Profil; Metriken zu Konsistenz.
- **UI‑Verbesserungen**: Claims + Legende, Wrapper‑Erklärung, Frage‑Preview, Auto‑Refresh (Progress), Tokens/Modelle **aus UI entfernt**.

---

## Profile (NO_SEARCH, SEARCH_AUTO, GOOGLE_OVERVIEW)

- **NO_SEARCH** (`*_NO_SEARCH`)  
  LLM antwortet ohne Websuche auf Basis von **Modellwissen**. `citations` bleiben i. d. R. **leer** (`[]`) bzw. verweisen nur auf modellinterne Wissenselemente.
- **SEARCH_AUTO** (`*_SEARCH_AUTO`)  
  Das LLM recherchiert (z. B. Google/Gemini‑Search), sammelt **Evidence** (Titel, URL, Domain, Datum/Snippet), fasst faktenbasiert zusammen.
- **GOOGLE_OVERVIEW**  
  Kompakter Überblick mit Quellenlisten („Top‑Treffer“) zur schnellen Orientierung.

> Das **NO‑Search‑Profil** spiegelt strukturell das **Search‑Profil**: gleiche Output‑Schema, aber **leere `citations`**, sofern keine Modell‑Quellen vorliegen.

---

## KPIs & Definitionen

**Alter (`age_days`)**  
Tage seit `published_at` der Quelle bis „jetzt“.

**Freshness‑Bucket**  
`today`, `≤7d`, `≤30d`, `≤90d`, `≤365d`, `>365d` (per Evidence).

**Freshness‑Index**  
\( \text{avg}( e^{-(\text{age\_days}/90)} ) \) über alle Evidenzen (Skala 0..1, **höher = aktueller**).

**Sentiment‑Score**  
Kontinuierlich in \([-1, +1]\).

**Sentiment‑Label**  
Schwellen: \(\le -0{,}2\) = **negativ**, \(\ge +0{,}2\) = **positiv**, sonst **neutral**.

**Visibility (0..1)**  
„**Wie prominent** steht die Marke im Antworttext im Fokus?“

**Inclusion (Accepted/Rejected)**  
Ob die Antwort thematisch **einzahlt** (Accepted) oder nicht (Rejected).

---

## Answer Quality (quality_flags) & Scores

**Ziel**: Neben *was* gesagt wird, prüfen **Qualität** & **Korrektheit**.

**Enum `aspect_scores.quality_flags`** (max. 5, nur Codes, konservativ flaggen):
- `HALLUCINATION_SUSPECTED` – Tatsachenbehauptung ohne Evidenz oder gegen Evidenz.
- `CONFUSES_WITH_OTHER_BRAND` – Verwechslung mit Namesakes.
- `OUTDATED_INFO` – Zeitkritisches veraltet **oder** Median `age_days` > 365 als „aktuell“.
- `MISSING_KEY_ASPECTS` – Offensichtliche Kernaspekte fehlen (z. B. Preis/Regulierung/Sicherheit bei entsprechender Frage).
- `UNSUPPORTED_SUPERLATIVES` – Superlative/Absolute ohne Evidenz.
- `INCOHERENT_OR_OFF_TOPIC` – Widersprüchlich/off‑topic.
- `SOURCE_BIAS_RISK` – Einseitige Quellenlage (z. B. >70 % Corporate/PR) ohne Gegencheck.
- `DATA_MISMATCH` – Zahlen/Daten inkonsistent (intern vs. Evidenz).
- `REGION_CONTEXT_MISMATCH` – Falscher Markt/Region/Zeitrahmen.
- `OTHER_QUALITY_ISSUE` – nur bei klar gravierendem Sonderfall.

**Quality‑Score (0..1)** — pro Run:  
Gewichtete Flags → **Risk** wird aufsummiert und auf max. **1.0** gekappt:  
`quality_risk_index = min(1.0, Summe(Gewichte der eindeutigen Flags))`  
`quality_score = 1 - quality_risk_index`

**Default‑Gewichte** (konservativ):
- HALLUCINATION_SUSPECTED **1.0**; CONFUSES_WITH_OTHER_BRAND **0.9**;  
- OUTDATED_INFO **0.7**; INCOHERENT_OR_OFF_TOPIC **0.7**; DATA_MISMATCH **0.7**;  
- MISSING_KEY_ASPECTS **0.6**; REGION_CONTEXT_MISMATCH **0.6**;  
- UNSUPPORTED_SUPERLATIVES **0.5**; SOURCE_BIAS_RISK **0.5**;  
- OTHER_QUALITY_ISSUE **0.3**.

> Anpassungen der Gewichte sind möglich; Standard ist auf **Strenge/Verlässlichkeit** getrimmt.

---

## Stabilitätsmetriken (Multi-Runs)

Bei **num_runs ≥ 2** je (Frage/Profil/Sprache/Stakeholder/Kategorie/Intent/Variante):

- **Agreement‑Rate (Sentiment)**  
  Anteil der Runs mit **gleichem** `sentiment_label` (negativ/neutral/positiv).

- **Narrative‑Overlap** (Jaccard, Ø paarweise)  
  Jaccard‑Index über Sets aus `aspect_scores.narrative`.

- **Risk‑Overlap** (Jaccard, Ø paarweise)  
  Jaccard‑Index über Sets aus `aspect_scores.risk_flags`.

- **Quality‑Overlap** (Jaccard, Ø paarweise)  
  **Neu:** Jaccard‑Index über Sets aus `aspect_scores.quality_flags`  
  (1.0 = sehr stabil; 0.0 = inkonsistent).

- **Source‑Overlap** (Jaccard, Ø paarweise)  
  Jaccard‑Index über **Domains** der Evidenz je Run.

- **Durchschnittliche Quality‑Indizes**  
  `avg_quality_risk_index`, `avg_quality_score` über die Runs.

---

## Evidence‑Enrichment & Domain‑Typisierung

Jede Evidence wird **normalisiert** und angereichert:
- `domain` (Normalisierung), `domain_type` (aus **CSV‑Seeds**),  
- `published_at`, `age_days`, `freshness_bucket`,  
- `original` (komplettes Originalobjekt unverändert).

**Konfigurationsdatei:** `domain_type_prompt.csv`  
Spalten: `domain_type`, `tld_hints`, `keyword_hints`, `example_domains`  
(vordefiniert für **CORPORATE, NEWS_MEDIA, TRADE_PROFESSIONAL, GOV_PUBLIC, NGO_ASSOCIATION, RESEARCH_EDUCATION, REVIEW_PLATFORM, SOCIAL_PLATFORM, BLOG_FORUM, OTHER_UNKNOWN**).

---

## Stakeholder‑Bibliothek & Prompt‑Prefixing

Datei **`stakeholder_library.xlsx`**:
- Sheet `map` (DE‑UI‑Label → `stakeholder_id`),
- Sheets `de/en/fr/it/rm` mit `display`, `prefix_template` (z. B. DE: *„Ich bin ein {stakeholder}.“*).

**Sonderfall (UI‑Label):** „**Entscheidungsträger aus Politik und Verwaltung**“ ersetzt „Politischer Entscheider“.  
**Logik:** Falls in der Frage kein `<STAKEHOLDER>` steht und Stakeholder ≠ generic → **Prefix** aus `prefix_template` wird der Frage **vorangestellt**.

---

## Fragebibliothek (Excel) & Intents

**`ki_question_library.xlsx`**
- **Sprach‑Sheets:** `de/en/fr/it/rm`  
  + Spalte **`intent_desc`** (aus Intents‑Tabelle; 1–6).  
- **`Intents`‑Sheet:** Beschreibung der **Intent‑Zahlen** 1–6 in fünf Sprachen.

Beispiel‑Intents:  
1 Überblick · 2 Leistung/Nutzen · 3 Risiken/Kritik · 4 Anwendung/How‑to · 5 Wettbewerb/Markt · 6 Strategie/Ausblick

---

## Konfiguration & Laufzeit

### Voraussetzungen
- **Python 3.10+**
- `pip install -r requirements.txt`

### Umgebungsvariablen
- `OPENAI_API_KEY` — für OpenAI **Responses API** (GPT‑5/5.1 etc.)
- `GEMINI_API_KEY` — für Google **Generative Language API** (Gemini)
- **Wichtig für GPT‑5/5.1 (Responses API):**  
  **Keine** klassischen Sampling‑Parameter (kein `temperature`, `top_p`, etc.). **Modelle & Limits sind fix** im Code.

### Start
```bash
streamlit run streamlit_app.py
```

### Retries
- **Leichtgewichtige Retries (max 3)** für Pass‑A/B‑API‑Calls bei 429/5xx (Backoff).

---

## Ausgaben / Exporte

**Excel‑Export** enthält u. a.:
- **Runs** — alle Einzelruns
- **Normalized** — normalisierte Pass‑B‑Objekte inkl.  
  `aspect_scores.quality_flags`, `quality_risk_index`, `quality_score`
- **Evidence** — angereicherte Quellen (inkl. `original`)
- **Stability_Metrics** —  
  `num_runs`, `agreement_rate_sentiment`,  
  `jaccard_narrative_avg`, `jaccard_risk_flags_avg`,  
  `jaccard_quality_flags_avg`, `jaccard_sources_avg`,  
  `avg_quality_risk_index`, `avg_quality_score`, `top_sentiment_label`
- **RawAnswers** — Rohtexte + Meta
- **Config** — Laufkonfiguration

---

## Changelog (Auszug)

**2025‑11‑18 — Konsolidierter Build**
- Tokens/Modelle **aus UI entfernt**; feste Limits.
- **Legende** (KPIs, Profile), Claims im Header.
- **Wrapper‑Erklärung** (free_emulation vs. stabilized) in Sidebar.
- **Fragen‑Preview** (bis 5 Zeilen), **Auto‑Refresh** für Progress.
- **Stakeholder‑Bibliothek** & Prefixing (inkl. Umbenennung).
- **Intents**‑Tabelle & `intent_desc` in allen Sprachen.
- **Evidence‑Enrichment** (inkl. `original`), Domain‑Typisierung & Freshness.
- **Gemini‑Wrapper**: No‑Search & Search‑Auto (inkl. Zitate).
- **Pass‑B‑Prompt**: klare Definition **Visibility/Sentiment**, knappe **Narrative/Risks**, **Answer‑Quality** mit Enum/Kriterien.
- **Stabilität**: Sentiment‑Agreement, Narrative/Risk/Quality‑Jaccard, Source‑Jaccard, Quality‑Score (Run‑Level & Gruppe).
- **NO‑Search** spiegelt **Search** Schema (leere `citations` möglich).

---

## Grenzen & Hinweise

- **Antwortqualität** hängt von externen Modellen ab; Flags und Scores sind **Heuristiken** (konservativ gewählt).
- **Domain‑Typisierung** per CSV‑Seeds ist **best‑effort**; feingranulare Regeln können ergänzt werden.
- **Kein Temperature/Top‑p** bei GPT‑5/5.1 (Responses API) – Designentscheidung & API‑Vorgabe.
- **Fair Use** der Search‑APIs beachten (Kontingente, Latenzen).

---

Bei Fragen oder Änderungswünschen (z. B. Gewichte, Schwellen, weitere Domaintypen) bitte ein Issue eröffnen oder die Config‑Dateien anpassen.
