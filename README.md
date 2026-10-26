DataStory AI
A Streamlit app that uploads a CSV, runs automated statistical analysis (trends, correlations, anomalies, segments), and generates an interactive 4-chapter data story. Every finding includes a "Show Calculation" expander with the exact Python code used to compute it. Optional Groq/Llama 3.1 AI adds plain-English explanations. One-click Markdown, HTML, PDF, and JSON export. Works 100% offline without any API key.

Table of contents
Features
Architecture and workflow
AI / LLM integration
Analysis tools
Memory and session state
Streamlit application
Export system
Supported data formats
Project structure
Technologies
Installation
Configuration (secrets / .env)
Running the app
Usage
Limitations and requirements

1. Features
Automated 4-chapter story: Big Picture → Hidden Opportunities → Unexpected Anomalies → What You Should Do Next.
Dataset profile: row/column counts, dtypes, missing values, duplicates, quality score (A–F), semantic column classification (numeric, categorical, datetime, text, ID).
Statistical detection engines (pure Python, no LLM required):
Trends — Mann-Kendall test + linear regression on time-series columns.
Correlations — Pearson & Spearman on all numeric pairs with Benjamini-Hochberg FDR correction; robustness checks (Pearson vs Spearman gap, trimmed re-test).
Anomalies — IQR outliers, Z-score extremes, sudden changes (3σ jumps), quality issues (high missingness, duplicates, constant columns).
Segments — 1-D (single categorical) and 2-D (two categoricals) high/low performing subgroups by z-score.
Evidence-backed findings: every finding carries an `evidence` dict with exact statistics, p-values, effect sizes, bounds, and row indices.
"Show Calculation" — copy-paste the exact Python code snippet that produced the finding (profiling, IQR, Z-score, Mann-Kendall, correlation with BH correction, Kruskal-Wallis, segment discovery).
Optional AI explanations — Groq (Llama 3.1 8B Instant) generates 2–3 sentence plain-English summaries with one actionable suggestion. Falls back to template summaries when no API key is configured.
Interactive Plotly charts per finding (scatter, line, bar, box, heatmap) with automatic configuration.
Assumption checklist before analysis (datetime parsing, ID columns, high cardinality, potential target, row semantics).
Follow-up chat — ask natural-language questions about the data; answered via rule-based fallbacks or Groq.
Multi-format export — Markdown, HTML, PDF (WeasyPrint), JSON.
Statistical guardrails — BH correction, robustness notes, sample-size warnings, causation disclaimer shown prominently on findings.

2. Architecture and workflow
User
  |
Streamlit UI (app.py)
  |
SessionState (df, profile, findings, chat_history, assumptions, groq_client)
  |
Analysis Pipeline (app.run_analysis)
     |
     +-- detect_trends (analysis/trends.py)
     +-- detect_correlations (analysis/correlations.py)  
     +-- detect_categorical_numeric_relationships
     +-- detect_anomalies (analysis/anomalies.py)
     +-- detect_segments (analysis/segments.py)
     +-- generate_action_recommendations
     +-- add_ai_explanations (ai/groq_client.py)
  |
Story Rendering (story/chapters.py)
     |
     +-- render_chapter_tabs → render_chapter → render_finding_card
     |       (evidence formatting via story/evidence.py)
     |       (code snippets via analysis/code_snippets.py)
     +-- render_followup_chat
  |
Export (export.py) → Markdown / HTML / PDF / JSON

Module by module:

Module	What it does
app.py	Streamlit entry point: session init, sidebar, upload, preview, assumption checklist, analyze button, story tabs, follow-up chat, footer exports.
analysis/profile.py	profile_dataframe → shape, dtypes, missing, duplicates, quality score, semantic column lists, numeric/categorical summaries.
analysis/trends.py	detect_trends → Mann-Kendall + linregress per (datetime, numeric) pair; sequential trend fallback.
analysis/correlations.py	detect_correlations → all pairs, both methods, BH adjustment, keep best method per pair, robustness_check, sample_size_note. detect_categorical_numeric_relationships → Kruskal-Wallis per (numeric, categorical).
analysis/anomalies.py	detect_anomalies → IQR, Z-score, sudden changes, quality issues; dedup by (type, column, method).
analysis/segments.py	detect_segments → 1-D segments (z>1.5), 2-D segments (z>2); min group size 5.
analysis/guardrails.py	bh_adjust, format_p, robustness_check, sample_size_note, CAUSATION_NOTE — pure functions, unit-tested.
analysis/code_snippets.py	FINDING_CODE_MAP → exact Python templates per finding type; get_code_snippet().
ai/groq_client.py	GroqClient (OpenAI-compatible) with retry/backoff, fallback templates per finding type, follow-up prompt builder.
ai/prompts.py	Prompt templates: FINDING_PROMPT_TEMPLATE, FOLLOWUP_PROMPT_TEMPLATE, CHAPTER_SUMMARY_PROMPT_TEMPLATE, ANOMALY_PROMPT_TEMPLATE.
story/chapters.py	CHAPTERS config, render_chapter_tabs, render_big_picture, render_finding_card, render_finding_chart (Plotly), render_followup_chat.
story/evidence.py	format_finding_evidence (metric cards), format_evidence_table (DataFrame), format_row_evidence, create_evidence_download, render_evidence_expander.
utils/csv_loader.py	load_csv / load_csv_from_upload → chardet encoding, delimiter detection, header row detection, column cleaning; returns (DataFrame, metadata).
utils/fallback.py	generate_finding_summary, generate_chapter_summary, generate_executive_summary — template summaries when AI unavailable.
export.py	export_to_markdown, export_to_html (markdown2 or simple fallback), export_to_pdf (WeasyPrint, degrades gracefully), export_findings_json.
tests/test_analysis.py	Pytest fixtures + tests for profiler, anomalies, trends (Mann-Kendall), correlations, segments, code_snippets, export.

Where control flow lives
Streamlit drives the UI; the analysis pipeline is a plain function call (run_analysis) invoked when the user clicks "Analyze & Generate Story". No LangGraph or agent framework — the workflow is deterministic and fast. AI calls (Groq) happen after all findings are computed, in a loop over findings, and never block the core analysis.

3. AI / LLM integration
Provider / SDK: Groq via requests (OpenAI-compatible chat/completions endpoint).
Model: llama-3.1-8b-instant (free tier: 30 RPM, 14,400 req/day, 6k TPM). Configurable via GROQ_MODEL env/secret.
Client: GroqClient in ai/groq_client.py. is_configured() checks for api_key in Streamlit secrets → GROQ_API_KEY env → GROQ_PROVIDER/GROQ_BASE_URL/GROQ_MODEL env.
Retry: max_retries=3, retry_delay=1.0s with linear backoff. Timeouts at 30s.
Prompts (ai/prompts.py):
FINDING_PROMPT_TEMPLATE — rules: use ONLY provided evidence numbers, no hallucination, 2–3 sentences, plain English, one actionable suggestion, reference actual values.
FOLLOWUP_PROMPT_TEMPLATE — answer user question from dataset context + findings summary.
CHAPTER_SUMMARY_PROMPT_TEMPLATE — 3–4 sentence chapter executive summary.
ANOMALY_PROMPT_TEMPLATE — detailed anomaly explanation.
What is sent to Groq: finding dict (type, title, description, evidence), dataset profile summary (shape, quality score). Full DataFrame is never sent. Evidence JSON is serialized and embedded in the prompt.
Fallback: _fallback_explanation per finding type (trend, correlation, outlier_iqr, outlier_zscore, sudden_change, segment, segment_2d, categorical_numeric, quality_issue, generic) produces template text identical in structure to AI output. App works 100% offline.

4. Analysis tools
All implemented in analysis/ as pure functions (df, profile) → List[Dict finding]. No LLM involvement.

Profile tool (profile_dataframe) — builds comprehensive profile dict used by every downstream detector.

Trend tool (detect_trends) — for each datetime × numeric pair: sorts by time, converts time to seconds since start, runs mann_kendall_test (pure NumPy/SciPy) and scipy.stats.linregress. Keeps findings with p<0.05 and |r|>0.3. Evidence includes τ, p, slope, R², direction, % change, start/end values & dates. Chart: scatter with OLS trendline.

Correlation tool (detect_correlations) — itertools.combinations on numeric columns. For each pair: scipy.stats.pearsonr & spearmanr. Collect all p-values → bh_adjust (Benjamini-Hochberg). Keep pairs with p_adj≤0.05 and |r|≥0.4. Deduplicate by keeping stronger method per pair. robustness_check computes Pearson, Spearman, and trimmed Pearson (drop 5% most extreme by combined |z|). Adds caveats for Pearson/Spearman gap>0.3 or trimmed shift>0.3. sample_size_note for n<30. Evidence: r, method, p, p_adj, n_tests_corrected, sample_size, relationship_type, direction, pearson_r, spearman_r, caveats. Chart: scatter (OLS line only for Pearson without robustness warnings).

Categorical-Numeric tool (detect_categorical_numeric_relationships) — Kruskal-Wallis per (numeric, categorical) with 2–20 categories, min group size 3. Keeps p<0.01. Evidence: H, p, group_means, max/min group, difference_pct. Chart: box plot.

Anomaly tools (detect_anomalies):
IQR — Q1, Q3, 1.5×IQR bounds. Reports if outlier%≥0.5% or count≥3. Evidence: bounds, outlier_count, outlier_percentage, example_low/high, row_indices. Chart: box with outlier highlight.
Z-score — mean, std, threshold=3. Reports only if max_z>4 or count>5. Evidence: mean, std, max_z, outlier_count, row_indices.
Sudden changes — for each datetime×numeric: sorted, diff, pct_change. Threshold = mean|diff| + 3·std|diff|. Reports top 3 jumps. Evidence: dates, values, absolute & percent change, direction. Chart: line.
Quality issues — missing>50%, duplicates, constant columns. Evidence per issue type.

Segment tools (detect_segments):
1-D — groupby categorical on numeric; filter count≥5; z=(group_mean−overall_mean)/overall_std. Report top/bottom if |z|≥1.5. Evidence: segment_value, segment_mean, segment_count, overall_mean, z_score, pct_difference, example_row_indices. Chart: bar of group means.
2-D — cross two categoricals (combos≤50); groupby both; filter count≥5; z-score. Report top/bottom if |z|≥2. Evidence: segment_values dict, segment_mean, count, z_score, pct_difference. Chart: heatmap pivot.

Action generator (generate_action_recommendations) — converts high-signal findings into prioritized actions:
|r|>0.6 correlations → "Investigate X→Y Relationship" (high)
|z|>2 segments → "Target High/Low-Performing Segment" (medium)
IQR outliers >5% → "Review Outliers in X" (high)
High missingness → "Address Missing Data in X" (high)
Sudden changes → "Investigate Sudden Change on date" (medium)
Sorted by priority (high→medium→low), capped at 5.

Code snippets (analysis/code_snippets.py) — FINDING_CODE_MAP with 14 templates keyed by finding type. get_code_snippet(type) returns the template. Templates are standalone, runnable Python using pandas/numpy/scipy with comments. Rendered in "Show Calculation" expander.

5. Memory and session state
Held in st.session_state (in-memory, per browser session).

Keys:
df — current DataFrame (sampled to max_rows if larger).
profile — output of profile_dataframe.
findings — dict with keys: big_picture, opportunities, anomalies, actions.
analysis_done — bool flag.
groq_client — GroqClient instance (reused).
chat_history — list of {role, content, evidence?} for follow-up chat.
assumptions — dict keyed by assumption_id with {id, type, title, description, column, default, confirmed}.
assumptions_confirmed — bool gate before analysis.
sample_data_loaded — bool for sample dataset tracking.
max_rows, min_correlation — sidebar config values.
metadata — upload metadata (encoding, delimiter, header_row, rows, cols, warnings).

Reset: "Analyze New File" button deletes all keys and st.rerun(). New upload also resets.

6. Streamlit application
app.py — single-page layout with wide mode, custom CSS.

Sidebar:
Groq status (configured / not configured) with model name.
Analysis options: max_rows (1k–100k, default 10k), min_correlation (0.1–0.9, default 0.4).
Export buttons (Markdown, HTML, PDF) — enabled only after analysis_done.
About panel describing the 4 chapters and "Show Calculation" feature.

Main flow:
No df → render_upload_section (file uploader + 3 sample dataset buttons).
df loaded → render_data_preview (tabs: Sample Data, Column Info, Profile Summary) → assumption checklist (if not confirmed) → render_analyze_button.
analysis_done → render_story (executive summary + 4 chapter tabs + follow-up chat) → render_footer (JSON export, summary markdown, reset).

Upload handling:
load_csv_from_upload → (df, metadata). Sampling if len(df)>max_rows. Profile computed. Session keys initialized. Warnings surfaced.

Sample datasets (sample_data/):
tips.csv (244 rows) — restaurant tips.
subscription_churn.csv (150 rows) — customer churn.
guardrail_stress.csv (180 rows) — designed to trigger spurious correlations for guardrail testing.

Assumption checklist — rendered after preview, before analyze button. Each assumption: checkbox with description. "All Confirmed" button enables analysis.

Chapter tabs — st.tabs with icons. Big Picture renders metrics, pie of semantic types, missing bar chart, interactive numeric/categorical distributions. Other chapters render finding cards via render_finding_card.

Finding card — expander with title. AI explanation (badge). Description. Guardrail caveats (warning box). Evidence as metric cards. Plotly chart. "Show Calculation" expander with syntax-highlighted code. "View Raw Evidence Table" expander with full evidence DataFrame.

Follow-up chat — st.chat_input. Rule-based router for mean/max/min, correlation, trend, anomaly questions. Falls back to GroqClient.explain_followup if configured. History persisted in session_state.chat_history.

7. Export system
export.py — four functions, all pure (findings, profile, metadata, df) → string/bytes.

export_to_markdown — structured report: header, dataset overview, executive summary, per-chapter findings with AI explanations, detailed findings with evidence tables, footer. No external deps.

export_to_html — wraps markdown via markdown2 (tables, fenced-code, header-ids) if available; else simple <br> fallback. Embeds full CSS (light theme, code blocks, ai-insight/evidence-item boxes). Self-contained single file.

export_to_pdf — calls export_to_html then WeasyPrint HTML(string=html).write_pdf(). Catches ImportError/OSError (missing GTK on Windows) and raises friendly ImportError suggesting HTML export + browser print-to-PDF.

export_findings_json — serializes {generated_at, profile, findings, metadata} with default=str for non-JSON types.

Sidebar buttons stream downloads with timestamped filenames: datastory_YYYYMMDD_HHMMSS.{md,html,pdf}. Footer button downloads datastory_findings.json.

8. Supported data formats
Defined in utils/csv_loader.py load_csv / load_csv_from_upload.

Format	Reader	Notes
.csv	pandas.read_csv	Encoding auto-detected (chardet, fallback latin-1/cp1252/iso-8859-1). Delimiter auto-detected from first 10 lines (, ; \t | :). Header row auto-detected (first line with delimiter, not starting with #). on_bad_lines='skip'. Duplicate column names renamed (col, col_1, col_2…).
.xlsx	Not supported	CSV only in current version.
.xls	Not supported	CSV only in current version.

Rejection cases: empty file, undecodable bytes, zero rows/columns after parse. Warnings collected in metadata['warnings'] and shown in UI.

9. Project structure
datastory/
├── app.py                      # Streamlit UI entry point
├── export.py                   # Markdown/HTML/PDF/JSON export
├── requirements.txt            # Dependencies
├── .gitignore                  # venv, __pycache__, .pyc, .env, .streamlit/secrets.toml
├── __init__.py                 # Package exports (version, analysis, ai, story, utils)
│
├── ai/
│   ├── __init__.py
│   ├── groq_client.py          # Groq API client + fallback templates
│   └── prompts.py              # All prompt templates
│
├── analysis/
│   ├── __init__.py
│   ├── profile.py              # Data profiling + quality scoring
│   ├── trends.py               # Mann-Kendall + linear regression trends
│   ├── correlations.py         # Correlation mining + BH correction + guardrails
│   ├── anomalies.py            # IQR, Z-score, sudden changes, quality issues
│   ├── segments.py             # 1-D & 2-D segment discovery
│   ├── guardrails.py           # BH adjust, format_p, robustness, sample note
│   └── code_snippets.py        # "Show Calculation" code templates
│
├── story/
│   ├── __init__.py
│   ├── chapters.py             # 4-chapter tabs, finding cards, charts, chat
│   └── evidence.py             # Evidence formatting, tables, downloads
│
├── utils/
│   ├── __init__.py
│   ├── csv_loader.py           # Robust CSV loading (encoding, delimiter, header)
│   └── fallback.py             # Template summaries (no AI)
│
├── tests/
│   ├── __init__.py
│   ├── test_analysis.py        # Unit tests for all detectors + export
│   └── test_guardrails.py      # Guardrail function tests
│
└── sample_data/
    ├── tips.csv
    ├── subscription_churn.csv
    └── guardrail_stress.csv

10. Technologies
Purpose	Library
UI	Streamlit ≥1.40
Data	Pandas ≥2.2, NumPy ≥1.26
Statistics	SciPy ≥1.12 (stats, norm, kruskal, pearsonr, spearmanr, linregress)
Charts	Plotly ≥5.20 (Express + Graph Objects)
AI (optional)	Groq via requests (OpenAI-compatible), Llama 3.1 8B Instant
Encoding detection	chardet ≥5.2
Markdown→HTML	markdown2 (optional)
PDF	WeasyPrint ≥60 (optional, needs GTK/Pango)
Testing	pytest ≥8.0

11. Installation
Requirements: Python 3.10+ (tested on 3.11/3.12). Groq API key optional.

```bash
cd datastory

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
# Optional for PDF export (Linux/macOS):
# pip install weasyprint
# Windows: WeasyPrint requires GTK — use HTML export + browser Print→PDF instead.
```

12. Configuration (secrets / .env)
Groq API key enables AI explanations. Two ways:

Streamlit secrets (recommended, works on Streamlit Cloud):
```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
[groq]
api_key = "your_groq_api_key_here"
model = "llama-3.1-8b-instant"      # optional
provider = "groq"                    # optional
base_url = "https://api.groq.com/openai/v1"  # optional
EOF
```

Environment variables (local dev):
```bash
export GROQ_API_KEY="your_groq_api_key_here"
# Optional overrides:
# export GROQ_MODEL="llama-3.1-8b-instant"
# export GROQ_PROVIDER="groq"
# export GROQ_BASE_URL="https://api.groq.com/openai/v1"
```

Create a key at https://console.groq.com/keys. Free tier is generous. Never commit secrets; .gitignore excludes .streamlit/secrets.toml and .env.

13. Running the app
```bash
streamlit run app.py
```
Open the printed URL (usually http://localhost:8501).

14. Usage
1. Launch the app. Sidebar shows Groq status (green if configured, yellow if not).
2. Upload a CSV via file uploader, OR click a sample dataset button (Tips / Subscription Churn / Guardrail Stress).
3. Review the Data Preview tabs (Sample Data, Column Info, Profile Summary).
4. Review the Assumption Checklist — confirm each detected assumption (datetime parsing, ID columns, high cardinality, potential target, row semantics).
5. Click **"Analyze & Generate Story"**.
6. Read the Executive Summary (expanded by default).
7. Explore the 4 chapter tabs:
   - **Big Picture** — metrics, type pie, missing bar, interactive distributions.
   - **Hidden Opportunities** — trends, correlations, categorical-numeric, segments.
   - **Unexpected Anomalies** — outliers, sudden changes, quality issues.
   - **What You Should Do Next** — prioritized actions linked to evidence.
8. Open any finding's **"Show Calculation"** expander to see the exact Python code.
9. Ask follow-up questions in the chat box (e.g., "What's the average of total_bill?", "Show correlation between total_bill and tip").
10. Export via sidebar (Markdown/HTML/PDF) or footer (JSON, Summary MD, New File).

15. Limitations and requirements
CSV only — Excel (.xlsx/.xls) not supported in current version.
Sampling — datasets larger than max_rows (default 10,000) are randomly sampled (seed=42) for speed; full data not analyzed.
No causal inference — all findings are associational. CAUSATION_NOTE ("Association only — this does not show that one variable causes the other.") is attached to every correlation finding.
Guardrails are advisory — BH correction controls FDR across tested pairs; robustness checks flag but do not suppress findings. User must interpret caveats.
Groq rate limits — free tier 30 RPM; very wide datasets (many numeric columns → many pairs) may hit limits. AI explanations are batched per finding with retry; failures fall back to templates.
PDF export on Windows — WeasyPrint requires GTK/Pango/Cairo which are non-trivial on Windows. Use HTML export → browser "Print to PDF" for equivalent output.
Semantic type inference — heuristic rules (uniqueness ratios, parse attempts); may misclassify edge cases. Assumption checklist lets user override.
No persistence — session state lives in browser memory; closing tab loses everything. Use exports to save.
Single-file upload — no multi-file joins or relational analysis.
Datetime parsing — pd.to_datetime with errors='coerce'; mixed formats may produce NaT rows silently dropped.
Minimum group sizes — segments require n≥5 per group; correlations require n≥10 per pair; trends require n≥10 per series.
Code snippets are templates — they reference column names from the finding evidence; copy-paste into a notebook with your DataFrame named `df` to reproduce.
