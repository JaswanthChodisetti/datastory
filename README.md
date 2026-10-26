# DataStory: Automated Data Storytelling Agent

A Streamlit app that turns any CSV/Excel dataset into a structured, evidence-backed data story with statistical rigor. Upload a file → get a 4-chapter narrative (Overview → Trends → Relationships → Anomalies) with interactive Plotly charts, "Show Calculation" expanders with exact Python code, statistical guardrails (Benjamini-Hochberg correction, robustness checks), optional AI explanations via Groq (Llama-3.1-8B), and one-click export to Markdown/HTML/PDF/JSON. Works **100% offline** — AI is purely optional.

---

## Table of Contents

- [Features](#features)
- [Architecture and Workflow](#architecture-and-workflow)
- [AI / LLM Integration](#ai--llm-integration)
- [Analysis Modules](#analysis-modules)
- [Statistical Guardrails](#statistical-guardrails)
- [Code Snippets / Show Calculation](#code-snippets--show-calculation)
- [Export](#export)
- [Supported Data Formats](#supported-data-formats)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Installation](#installation)
- [Configuration (Optional AI)](#configuration-optional-ai)
- [Running the App](#running-the-app)
- [Usage](#usage)
- [Sample Datasets](#sample-datasets)
- [Testing](#testing)
- [Limitations and Requirements](#limitations-and-requirements)

---

## Features

| Category | Capabilities |
|----------|--------------|
| **Data Ingestion** | Robust CSV/Excel loading with auto-detection of encoding, delimiter, and header row |
| **Profiling** | Shape, dtypes, missingness, duplicate rows, quality score (completeness, uniqueness, consistency, validity) |
| **Trend Detection** | Mann-Kendall τ + Sen's slope for monotonic trends; linear regression on datetime & sequential indices |
| **Correlations** | Pearson & Spearman with Benjamini-Hochberg FDR correction; Kruskal-Wallis for categorical-numeric pairs |
| **Anomaly Detection** | IQR (1.5×), Z-score (3σ), sudden time-series jumps, data quality flags (missing, duplicates, constant cols) |
| **Segment Discovery** | 1D: high-variance segments via z-score vs overall mean; 2D: cross-tab heatmaps with lift & sample size |
| **Statistical Guardrails** | BH multiple-testing correction, effect-size thresholds, minimum sample-size gates, robustness checks (Spearman vs Pearson, bootstrap CI) |
| **AI Explanations** | Optional Groq (Llama-3.1-8B-instant) for natural-language finding summaries; template fallback when offline |
| **Show Calculation** | Every finding card has an expander with the exact Python code used to compute it |
| **Interactive Charts** | Plotly charts (line, bar, scatter, heatmap, box) with hover, zoom, download |
| **Follow-up Chat** | Ask questions about findings; context-aware answers grounded in evidence |
| **Export** | Markdown, HTML, PDF (via WeasyPrint), JSON — all findings, charts, and code snippets |
| **Testing** | Pytest suite covering analysis modules + guardrails regression tests |

---

## Architecture and Workflow

```
User
  │
  ▼
Streamlit UI (app.py)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Data Pipeline                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  Load    │─▶│ Profile  │─▶│  Assumptions          │     │
│  │  CSV/    │  │  Data    │  │  Checklist          │     │
│  │  Excel   │  │  Quality │  │  (user confirms)    │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  4-Chapter Analysis (analysis/ modules)                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────┐  │
│  │  Chapter 1  │ │  Chapter 2  │ │  Chapter 3  │ │ Chap 4│  │
│  │  Overview   │ │  Trends     │ │Relationships│ │Anomalies│  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────┘  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Guardrails Layer (analysis/guardrails.py)                  │
│  • Benjamini-Hochberg FDR correction                        │
│  • Effect-size thresholds (|r| ≥ 0.3, η² ≥ 0.06)           │
│  • Minimum sample size per segment (n ≥ 30)                │
│  • Robustness: Spearman vs Pearson, bootstrap CI (n=1000)  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Story Rendering (story/chapters.py)                        │
│  • Finding cards with Plotly charts                         │
│  • "Show Calculation" expanders (code_snippets.py)          │
│  • Evidence table with download                             │
│  • Follow-up chat (AI or template fallback)                 │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Export (export.py)                                         │
│  Markdown  •  HTML  •  PDF (WeasyPrint)  •  JSON           │
└─────────────────────────────────────────────────────────────┘
```

### Module-by-Module Flow (`app.py`)

| Step | Module | Purpose |
|------|--------|---------|
| 1 | `utils/csv_loader.py` | Load file → DataFrame with encoding/delimiter detection |
| 2 | `analysis/profile.py` | Compute profile: shape, types, missing, quality score |
| 3 | `app.py` (UI) | Show assumption checklist (user confirms/edits) |
| 4 | `analysis/trends.py` | Mann-Kendall + linear regression on datetime/sequential cols |
| 5 | `analysis/correlations.py` | Pearson/Spearman + Kruskal-Wallis with BH correction |
| 6 | `analysis/anomalies.py` | IQR, Z-score, sudden changes, quality flags |
| 7 | `analysis/segments.py` | 1D & 2D high-variance segment discovery |
| 8 | `analysis/guardrails.py` | Filter findings: BH, effect size, sample size, robustness |
| 9 | `story/chapters.py` | Render 4 chapters: findings + charts + code snippets |
| 10 | `ai/groq_client.py` | Optional: generate natural-language explanations |
| 11 | `story/evidence.py` | Format evidence table for download |
| 12 | `export.py` | Export story to Markdown/HTML/PDF/JSON |

---

## AI / LLM Integration

| Aspect | Details |
|--------|---------|
| **Provider** | Groq (OpenAI-compatible API) |
| **Model** | `llama-3.1-8b-instant` (default, configurable) |
| **Client** | `ai/groq_client.py` — retry with exponential backoff (3 attempts), fallback to template summaries |
| **Prompts** | `ai/prompts.py` — structured templates per finding type (trend, correlation, anomaly, segment) |
| **What's Sent** | Finding metadata + statistics (never raw data); max ~2KB per call |
| **Fallback** | `utils/fallback.py` — deterministic template summaries when API key missing or rate-limited |
| **Cost** | Free tier: 30 RPM, 14,400 requests/day |

**The app works 100% without an API key.** AI only adds natural-language narratives; all calculations, charts, and guardrails are deterministic Python.

---

## Analysis Modules

### 1. Profiling (`analysis/profile.py`)
- **Shape**: rows × columns
- **Column types**: numeric, categorical, datetime, text (auto-inferred)
- **Missingness**: per-column count & percentage
- **Duplicates**: exact duplicate row count
- **Quality Score** (0–100): weighted composite of
  - Completeness (30%): non-null ratio
  - Uniqueness (20%): 1 - duplicate ratio
  - Consistency (25%): type conformity, datetime parse success
  - Validity (25%): range checks (e.g., % in [0,100], no negative ages)

### 2. Trends (`analysis/trends.py`)
- **Mann-Kendall τ**: non-parametric monotonic trend test (p-value, τ)
- **Sen's Slope**: robust slope estimate (median of pairwise slopes)
- **Linear Regression**: OLS on datetime (converted to ordinal) or sequential index
- **Output**: direction (increasing/decreasing/flat), magnitude, p-value, R²
- **Guardrail**: only reported if p < 0.05 after BH correction AND |slope| > min_effect

### 3. Correlations (`analysis/correlations.py`)
- **Numeric-Numeric**: Pearson *r* + Spearman ρ with p-values
- **Categorical-Numeric**: Kruskal-Wallis H-test (η² effect size)
- **Benjamini-Hochberg**: FDR correction across all tested pairs (α = 0.05)
- **Output**: significant pairs with effect size, CI, corrected p-value
- **Guardrail**: |r| ≥ 0.3, η² ≥ 0.06, n ≥ 30 per group

### 4. Anomalies (`analysis/anomalies.py`)
| Method | Threshold | Applies To |
|--------|-----------|------------|
| IQR | 1.5 × IQR below Q1 / above Q3 | Numeric columns |
| Z-score | |z| > 3 | Numeric columns |
| Sudden Change | |Δ| > 3 × rolling_std (window=5) | Datetime-indexed series |
| Quality Flags | Missing > 50%, constant cols, duplicate rows | All columns |

### 5. Segments (`analysis/segments.py`)
- **1D Segments**: For each categorical column, compute metric (mean of target numeric) per category; flag categories where |z| > 2 vs overall mean; require n ≥ 30
- **2D Segments**: Cross-tab of two categorical columns vs target numeric; heatmap with lift = segment_mean / overall_mean; filter by sample size & effect size

### 6. Guardrails (`analysis/guardrails.py`)
```python
# Core filters applied to every finding
def apply_guardrails(findings, profile):
    # 1. Benjamini-Hochberg FDR (alpha=0.05)
    # 2. Minimum effect size thresholds
    # 3. Minimum sample size (n >= 30 per group)
    # 4. Robustness checks:
    #    - Pearson vs Spearman sign agreement
    #    - Bootstrap 95% CI (1000 resamples) excludes 0
    # 5. Deduplication: merge overlapping findings
    return filtered_findings
```

---

## Code Snippets / Show Calculation

Every finding card in the UI has a **"Show Calculation"** expander with the exact, runnable Python code that produced the result. Generated from `analysis/code_snippets.py`.

**Example — Mann-Kendall Trend:**
```python
# Show Calculation for: "Revenue shows increasing trend (τ=0.42, p=0.003)"
import numpy as np
from scipy.stats import kendalltau

# Data preparation
dates = pd.to_datetime(df['date']).sort_values()
revenue = df.loc[dates.index, 'revenue'].values
x = np.arange(len(revenue))  # sequential index

# Mann-Kendall test
tau, p_value = kendalltau(x, revenue)

# Sen's slope (robust)
slopes = [(revenue[j] - revenue[i]) / (j - i) 
          for i in range(len(revenue)) 
          for j in range(i+1, len(revenue))]
sen_slope = np.median(slopes)

print(f"τ = {tau:.3f}, p = {p_value:.4f}, Sen's slope = {sen_slope:.2f}")
```

**Example — Correlation with BH Correction:**
```python
# Show Calculation for: "price ↔ quantity: r=-0.61 (p_adj=0.001)"
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

pairs = [('price', 'quantity'), ('price', 'discount'), ...]
p_values = []
results = {}

for a, b in pairs:
    r_pearson, p_pearson = pearsonr(df[a], df[b])
    r_spearman, p_spearman = spearmanr(df[a], df[b])
    results[(a,b)] = {'pearson': r_pearson, 'spearman': r_spearman, 
                       'p_pearson': p_pearson, 'p_spearman': p_spearman}
    p_values.append(p_pearson)

# Benjamini-Hochberg correction
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

## Export

| Format | Command | Dependencies |
|--------|---------|--------------|
| **Markdown** | `export.to_markdown(story)` | Built-in |
| **HTML** | `export.to_html(story)` | Built-in (includes embedded Plotly.js) |
| **PDF** | `export.to_pdf(story)` | `weasyprint` + system fonts |
| **JSON** | `export.to_json(story)` | Built-in |

**PDF Notes**: Requires WeasyPrint system dependencies (`cairo`, `pango`, `gdk-pixbuf`). On Ubuntu: `sudo apt-get install libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0`.

---

## Supported Data Formats

| Extension | Reader | Notes |
|-----------|--------|-------|
| `.csv` | `pandas.read_csv` | Auto-detects encoding (utf-8, latin-1, cp1252), delimiter (`,`, `;`, `\t`, `|`), header row |
| `.xlsx` | `pandas.read_excel` | Requires `openpyxl`; reads first sheet |
| `.xls` | `pandas.read_excel` | Requires `xlrd`; reads first sheet |

**Validation**: Rejects empty files, files with 0 rows or 0 columns, and files where all columns are constant.

---

## Project Structure

```
datastory/
├── app.py                      # Main Streamlit app entry point
├── export.py                   # Markdown, HTML, PDF, JSON export
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore
├── sample_data/
│   ├── tips.csv                # Classic seaborn tips dataset (244 rows)
│   ├── subscription_churn.csv  # Synthetic subscription data with churn
│   └── guardrail_stress.csv    # Stress test: outliers, missing, duplicates
├── __init__.py                 # Package exports
├── analysis/
│   ├── __init__.py
│   ├── profile.py              # Data profiling & quality scoring
│   ├── trends.py               # Mann-Kendall, linear regression trends
│   ├── correlations.py         # Pearson/Spearman + Kruskal-Wallis
│   ├── anomalies.py            # IQR, Z-score, sudden changes, quality flags
│   ├── segments.py             # 1D & 2D high-variance segment discovery
│   ├── guardrails.py           # Statistical guardrails (BH, robustness)
│   └── code_snippets.py        # "Show Calculation" exact code templates
├── ai/
│   ├── __init__.py
│   ├── groq_client.py          # Groq API client with retry/fallback
│   └── prompts.py              # Structured prompt templates
├── story/
│   ├── __init__.py
│   ├── chapters.py             # 4-chapter rendering, finding cards, Plotly charts, follow-up chat
│   └── evidence.py             # Evidence formatting & download
├── utils/
│   ├── __init__.py
│   ├── csv_loader.py           # Robust CSV loading (encoding, delimiter, header)
│   └── fallback.py             # Template-based summaries when AI unavailable
└── tests/
    ├── __init__.py
    ├── test_analysis.py        # Pytest suite for analysis modules
    └── test_guardrails.py      # Guardrails + regression tests for segment bugs
```

---

## Technologies

| Purpose | Library |
|---------|---------|
| UI | Streamlit |
| LLM (optional) | Groq (OpenAI-compatible) — Llama-3.1-8B-instant |
| Data Handling | Pandas, NumPy |
| Statistics | SciPy, StatsModels, scikit-learn |
| Charts (Interactive) | Plotly Express / Graph Objects |
| Charts (Static/PDF) | Matplotlib, Seaborn |
| PDF Export | WeasyPrint |
| CSV Loading | charset-normalizer (encoding), python-sniffer (delimiter) |
| Config | python-dotenv |
| Testing | pytest |

---

## Installation

**Requirements**: Python 3.10+ (developed on 3.12)

```bash
cd /mnt/d/hackthon/datastory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```text
streamlit>=1.35
pandas>=2.2
numpy>=1.26
scipy>=1.13
statsmodels>=0.14
scikit-learn>=1.4
plotly>=5.22
matplotlib>=3.8
seaborn>=0.13
weasyprint>=61
openpyxl>=3.1
xlrd>=2.0
charset-normalizer>=3.3
python-dotenv>=1.0
groq>=0.4
pytest>=8.2
```

---

## Configuration (Optional AI)

Create `.streamlit/secrets.toml` in the project root (or set environment variables):

```toml
[groq]
api_key = "gsk_..."          # Get free key at console.groq.com
# provider = "groq"          # Optional: override provider
# base_url = "https://api.groq.com/openai/v1"
# model = "llama-3.1-8b-instant"
```

**Without this config**, the app runs fully offline using template-based summaries (`utils/fallback.py`).

---

## Running the App

```bash
cd /mnt/d/hackthon/datastory
source .venv/bin/activate    # if not already active

# Run from parent directory (so `datastory` is importable as package)
python -m streamlit run datastory/app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. **Upload Data** — Drag & drop a CSV or Excel file (or pick a sample dataset)
2. **Review Assumptions** — Confirm/fix column types, missing handling, date parsing
3. **Read the Story** — Four chapters render automatically:
   - **Chapter 1: Overview** — Shape, types, missingness, quality score, key stats
   - **Chapter 2: Trends** — Significant trends with Mann-Kendall + regression charts
   - **Chapter 3: Relationships** — Correlation matrix, cat-num effects, segment heatmaps
   - **Chapter 4: Anomalies** — Outliers, sudden changes, quality flags
4. **Interact** — Hover charts, expand "Show Calculation" for code, download evidence CSV
5. **Ask Follow-ups** — Chat box at bottom: "Why did revenue dip in March?" "Show segments by region"
6. **Export** — Top-right: Download as Markdown, HTML, PDF, or JSON

---

## Sample Datasets

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `sample_data/tips.csv` | 244 | 7 | Classic seaborn tips: total_bill, tip, sex, smoker, day, time, size |
| `sample_data/subscription_churn.csv` | ~2,000 | 12 | Synthetic SaaS: tenure, monthly_charges, contract, churn, etc. |
| `sample_data/guardrail_stress.csv` | ~500 | 10 | Stress test: outliers, 30% missing, duplicates, constant cols |

Load via the "Use Sample Data" dropdown in the uploader.

---

## Testing

```bash
cd /mnt/d/hackthon/datastory
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_analysis.py -v
pytest tests/test_guardrails.py -v
```

### Test Coverage
- `test_analysis.py`: Profile, trends, correlations, anomalies, segments — happy path + edge cases
- `test_guardrails.py`: BH correction, effect-size filtering, sample-size gates, robustness checks, segment deduplication regression

---

## Limitations and Requirements

| Limitation | Details |
|------------|---------|
| **Memory** | Entire DataFrame held in memory; recommended < 100k rows for responsive UI |
| **PDF Export** | Requires system libraries (`cairo`, `pango`, `gdk-pixbuf`); may need `sudo apt install` on Linux |
| **AI Rate Limits** | Groq free tier: 30 RPM; app caches explanations per session |
| **Datetime Parsing** | Auto-detects common formats; ambiguous formats (MM/DD vs DD/MM) may need manual correction in assumptions |
| **Categorical Cardinality** | High-cardinality categorical columns (>50 unique) are treated as text, not segments |
| **No Model Training** | This is descriptive analytics only — no ML model building or prediction |
| **Single Table** | One dataset per session; no joins or multi-table analysis |

---

## License

MIT License — see `LICENSE` file (add one if needed).

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Add tests for new analysis types or guardrails
4. Ensure `pytest tests/ -v` passes
5. Submit a PR

---

**Built for analysts who want statistical rigor without the boilerplate.** 📊
