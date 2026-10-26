# DataStory AI — MVP

> **Every number has a receipt.** DataStory AI turns raw CSV files into a 4-chapter data story with statistical evidence and "Show Calculation" expanders revealing the exact Python code behind every finding.

Built for the **Infinity AI Forge Hackathon**.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **📊 4-Chapter Story** | Big Picture → Hidden Opportunities → Unexpected Anomalies → Recommended Actions |
| **🔍 Show Calculation** | Every finding has an expander with the exact Python code used to compute it |
| **📈 Interactive Charts** | Plotly-powered visualizations (hover, zoom, download) |
| **🤖 Optional AI Narrative** | Groq API (free tier: Llama 3.1 8B Instant) for plain-English explanations |
| **🛡️ Offline-First** | Core analysis works 100% without internet or API keys |
| **📤 Export Reports** | Markdown, HTML, PDF, and JSON export |
| **⚡ Assumption Checklist** | Validates data before analysis (datetime cols, ID cols, cardinality, target heuristic) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd /mnt/d/hackthon/datastory
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
datastory/
├── app.py                      # Main Streamlit app entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── analysis/
│   ├── __init__.py
│   ├── profiler.py             # Data profiling (shape, types, missing, quality score)
│   ├── anomalies.py            # IQR, Z-score, sudden change, quality issues
│   ├── trends.py               # Mann-Kendall, linear regression, sequential trends
│   ├── correlations.py         # Pearson/Spearman, categorical-numeric (Kruskal-Wallis)
│   ├── segments.py             # 1D & 2D segment discovery
│   └── code_snippets.py        # "Show Calculation" code templates
├── story/
│   ├── __init__.py
│   └── chapters.py             # 4-chapter rendering, finding cards, charts
├── ai/
│   ├── __init__.py
│   ├── groq_client.py          # Groq API client (OpenAI-compatible)
│   └── prompts.py              # Prompt templates
├── export.py                   # Markdown, HTML, PDF, JSON export
├── sample_data/
│   ├── tips.csv                # Classic tips dataset
│   ├── subscription_churn.csv  # Subscription data with churn labels
│   └── guardrail_stress.csv    # Stress test data (outliers, missing, duplicates)
└── tests/
    ├── __init__.py
    └── test_analysis.py        # Pytest suite for analysis modules
```

---

## 📊 The 4 Chapters

| Chapter | Focus | Key Analyses |
|---------|-------|--------------|
| **1. Big Picture** | Data overview | Shape, types, missingness, quality score, distributions |
| **2. Hidden Opportunities** | Trends & relationships | Mann-Kendall trends, Pearson/Spearman correlations, Kruskal-Wallis segments |
| **3. Unexpected Anomalies** | Outliers & quality | IQR/Z-score outliers, sudden changes, duplicates, high missingness |
| **4. Recommended Actions** | What to do next | Prioritized actions with evidence from Chapters 1-3 |

---

## 🔬 Statistical Methods Used

| Analysis | Method | Significance |
|----------|--------|--------------|
| Outliers (IQR) | Tukey's fences (1.5×IQR) | Non-parametric, robust |
| Outliers (Z-score) | \|Z\| > 3 | Parametric, normal assumption |
| Trends (time) | Mann-Kendall τ + Linear Regression | Non-parametric trend test |
| Trends (sequence) | Mann-Kendall τ | Row-order trends |
| Correlations | Pearson (linear) + Spearman (monotonic) | p < 0.05, \|r\| ≥ 0.4 |
| Cat-Num | Kruskal-Wallis H-test | p < 0.01, non-parametric ANOVA |
| Segments (1D) | Z-score vs overall mean | \|z\| ≥ 1.5, n ≥ 5 |
| Segments (2D) | Z-score on cross-tabs | \|z\| ≥ 2.0, n ≥ 5 |

---

## 🤖 AI Configuration (Optional)

Add your Groq API key to enable AI explanations:

### Streamlit Secrets (`.streamlit/secrets.toml`)
```toml
[groq]
api_key = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
# Optional overrides:
# provider = "groq"
# base_url = "https://api.groq.com/openai/v1"
# model = "llama-3.1-8b-instant"
```

### Environment Variable
```bash
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
streamlit run app.py
```

**Free Tier Limits:** 30 RPM, 14,400 requests/day, 6,000 TPM

---

## 📦 Sample Datasets

| File | Rows | Description |
|------|------|-------------|
| `tips.csv` | 244 | Classic seaborn tips dataset |
| `subscription_churn.csv` | 100 | Synthetic subscription data with churn |
| `guardrail_stress.csv` | 140 | Stress test: outliers, missing, duplicates, constant columns |

Load them from the sidebar "Sample Datasets" dropdown.

---

## 🧪 Running Tests

```bash
cd /mnt/d/hackthon/datastory
pytest tests/ -v
```

---

## 📤 Export Formats

| Format | Function | Use Case |
|--------|----------|----------|
| Markdown | `export_to_markdown()` | GitHub, Notion, Obsidian |
| HTML | `export_to_html()` | Web viewing, email |
| PDF | `export_to_pdf()` | Formal reports (requires `weasyprint`) |
| JSON | `export_findings_json()` | Programmatic use, APIs |

---

## 🛠️ Development

### Adding a New Finding Type

1. Add detection logic in `analysis/` modules
2. Add code template in `analysis/code_snippets.py`
3. Register in `FINDING_CODE_MAP`
4. Findings automatically get "Show Calculation" expander

### Dependencies

Core (required):
- `streamlit` — Web UI
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `plotly` — Interactive charts
- `scipy` — Statistical tests
- `ydata-profiling` — Automated profiling

Optional:
- `groq` / `requests` — AI explanations
- `markdown2` — Markdown→HTML
- `weasyprint` — PDF export

---

## 🎯 Hackathon Submission Checklist

- [x] CSV upload & parsing
- [x] Statistical profiling & anomaly detection
- [x] Interactive Plotly charts
- [x] 4 story chapters with evidence-backed findings
- [x] "Show Calculation" expanders (core differentiator)
- [x] Markdown/HTML/PDF/JSON export
- [x] Optional Groq AI narrative (offline-first)
- [x] Assumption checklist UX
- [x] Sample datasets included
- [x] Basic test suite
- [x] README documentation

---

## 📜 License

MIT License — Built for Infinity AI Forge Hackathon 2024.

---

## 🙏 Acknowledgments

- **Groq** for free LLM API tier
- **Streamlit** for rapid UI framework
- **Plotly** for interactive visualizations
- **ydata-profiling** for automated EDA
- **scipy** for statistical rigor