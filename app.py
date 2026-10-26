"""DataStory AI - Main Streamlit Application.

Upload a CSV, get an interactive 4-chapter data story with evidence-backed findings.
Every finding includes a "Show Calculation" expander with the exact code used.
"""

import os
import sys

# Make the `datastory` package importable no matter where Streamlit is launched from.
# Streamlit only puts this file's own folder on sys.path, so add its parent folder too.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import streamlit as st
import pandas as pd
import json
import io
import html
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

# Local imports
from datastory.utils import load_csv_from_upload, get_column_info, generate_executive_summary
from datastory.analysis import (
    profile_dataframe,
    get_profile_summary_text,
    detect_trends,
    detect_correlations,
    detect_anomalies,
    detect_segments
)
from datastory.analysis.code_snippets import get_code_snippet
from datastory.ai import GroqClient
from datastory.story import (
    CHAPTERS,
    render_chapter_tabs
)
from datastory.export import export_to_markdown, export_to_html, export_to_pdf



# ---------------------------------------------------------------------------
# Emoji filter: strips emoji from any text this app (or its story/export
# modules) sends to Streamlit, so the UI stays clean without editing every file.
# ---------------------------------------------------------------------------
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u231A\u231B\u23E9-\u23FF\u2600-\u27BF\u2B00-\u2BFF"
    "\uFE0E\uFE0F\u200D\u20E3]+"
)


def strip_emoji(text):
    """Remove emoji from a string and tidy the spaces they leave behind."""
    if not isinstance(text, str) or not _EMOJI_RE.search(text):
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)           # double spaces left behind
    cleaned = re.sub(r"(?m)^[ \t]+(?=\S)", "", cleaned)     # leading space on each line
    cleaned = re.sub(r"(?m)^(#{1,6}) +", r"\1 ", cleaned)   # keep "## Title" markdown intact
    return cleaned.strip(" \t")


def _install_emoji_filter():
    if getattr(st, "_datastory_emoji_filter", False):
        return
    st._datastory_emoji_filter = True

    def wrap_first_arg(name):
        original = getattr(st, name, None)
        if original is None:
            return

        def wrapper(*args, **kwargs):
            if args:
                args = (strip_emoji(args[0]),) + args[1:]
            for key in ("label", "body", "help", "placeholder"):
                if key in kwargs:
                    kwargs[key] = strip_emoji(kwargs[key])
            return original(*args, **kwargs)

        wrapper.__name__ = name
        setattr(st, name, wrapper)

    for fn in ("markdown", "write", "header", "subheader", "title", "caption", "text",
               "button", "download_button", "expander", "success", "info", "warning",
               "error", "metric", "toast", "spinner", "checkbox", "selectbox",
               "radio", "popover"):
        wrap_first_arg(fn)

    original_tabs = st.tabs

    def tabs(labels, *args, **kwargs):
        return original_tabs([strip_emoji(l) for l in labels], *args, **kwargs)

    st.tabs = tabs


# Page configuration
st.set_page_config(
    page_title="DataStory AI",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded"
)

_install_emoji_filter()


# Custom CSS for better styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(139,92,246,.28), transparent 60%),
        radial-gradient(800px 500px at 95% 0%, rgba(6,182,212,.20), transparent 60%),
        radial-gradient(700px 500px at 50% 110%, rgba(236,72,153,.14), transparent 60%),
        #0b0d17;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1180px; }

/* Hero */
.hero { text-align: center; padding: 2.2rem 1rem 1.2rem; }
.hero-pill {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    background: rgba(139,92,246,.15); border: 1px solid rgba(139,92,246,.4);
    color: #c4b5fd; font-size: .8rem; font-weight: 600; letter-spacing: .5px;
    margin-bottom: 1rem;
}
.main-header {
    font-size: 3.4rem; font-weight: 800; letter-spacing: -1.5px; line-height: 1.1;
    background: linear-gradient(90deg, #a78bfa 0%, #22d3ee 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .6rem;
}
.sub-header { color: #9ba3c7; font-size: 1.1rem; max-width: 640px; margin: 0 auto 1.4rem; }
.hero-chips { display: flex; gap: .6rem; justify-content: center; flex-wrap: wrap; }
.chip {
    padding: .4rem .9rem; border-radius: 12px; font-size: .85rem; color: #d6d9f2;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
    backdrop-filter: blur(8px);
}

/* Glass cards */
.finding-card, .metric-card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    border-radius: 16px; padding: 1.2rem; margin: .7rem 0;
    backdrop-filter: blur(12px); box-shadow: 0 8px 30px rgba(0,0,0,.25);
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.finding-card:hover {
    transform: translateY(-3px); border-color: rgba(139,92,246,.6);
    box-shadow: 0 12px 40px rgba(139,92,246,.22);
}

/* Badges */
.evidence-badge, .ai-badge, .calculation-badge {
    padding: .22rem .65rem; border-radius: 999px; font-size: .75rem; font-weight: 600;
}
.evidence-badge { background: rgba(74,222,128,.14); color: #4ade80; border: 1px solid rgba(74,222,128,.3); }
.ai-badge { background: rgba(167,139,250,.14); color: #c4b5fd; border: 1px solid rgba(167,139,250,.3); }
.calculation-badge { background: rgba(34,211,238,.12); color: #22d3ee; border: 1px solid rgba(34,211,238,.3); font-family: monospace; }

/* Assumptions */
.assumption-item {
    background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.25);
    border-left: 4px solid #f59e0b; padding: .8rem 1rem; margin: .5rem 0; border-radius: 12px;
}
.assumption-confirmed { background: rgba(74,222,128,.1); border-color: rgba(74,222,128,.25); border-left-color: #4ade80; }

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(255,255,255,.04); padding: 6px; border-radius: 14px;
    border: 1px solid rgba(255,255,255,.08);
}
.stTabs [data-baseweb="tab"] {
    padding: .55rem 1.1rem; border-radius: 10px; background: transparent; color: #9ba3c7; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4); color: white !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; padding: .6rem 1rem;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); color: #eceefc;
    transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4); border-color: transparent;
    transform: translateY(-2px); box-shadow: 0 8px 24px rgba(139,92,246,.4); color: white;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4); border: none; color: white;
}

/* Upload dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(139,92,246,.06); border: 2px dashed rgba(139,92,246,.5);
    border-radius: 18px; padding: 2rem; transition: all .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(139,92,246,.12); border-color: #22d3ee; }

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    border-radius: 16px; padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* Expanders, sidebar, dataframes, alerts */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08); border-radius: 14px;
}
section[data-testid="stSidebar"] {
    background: rgba(18,21,42,.85); backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,.08); }
[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: rgba(255,255,255,.08) !important; }
h3 { font-weight: 700; letter-spacing: -.3px; }
</style>
""", unsafe_allow_html=True)

# Extra polish layered on top of the base theme above
st.markdown("""
<style>
.stApp, .stApp p, .stApp li, .stApp label, .stApp span { color: #e6e8f5; }
h1, h2, h3, h4 { color: #f4f5ff; }
::selection { background: rgba(139,92,246,.45); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,.35); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,.6); }

/* Subtle grid texture behind the content */
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(ellipse at 50% 0%, #000 20%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 0%, #000 20%, transparent 75%);
}

/* Hero refinements */
.hero-logo { display:flex; justify-content:center; margin-bottom: .9rem;
    filter: drop-shadow(0 8px 28px rgba(139,92,246,.55)); }
.hero-pill::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%;
    background:#a78bfa; margin-right:.5rem; box-shadow:0 0 10px #a78bfa; vertical-align:middle; }
.chip::before { content:""; display:inline-block; width:6px; height:6px; border-radius:50%;
    background:linear-gradient(120deg,#8b5cf6,#22d3ee); margin-right:.5rem; vertical-align:middle; }

/* Progress stepper */
.stepper { display:flex; justify-content:center; align-items:center; gap:.4rem; margin:.4rem 0 1.4rem; flex-wrap:wrap; }
.step { display:flex; align-items:center; gap:.5rem; padding:.35rem .8rem .35rem .4rem; border-radius:999px;
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); color:#8f98bf; font-size:.82rem; font-weight:600; }
.step .num { width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
    font-size:.72rem; background:rgba(255,255,255,.08); color:#aab1d6; }
.step.active { color:#fff; border-color:rgba(139,92,246,.6); background:rgba(139,92,246,.14); }
.step.active .num { background:linear-gradient(120deg,#8b5cf6,#06b6d4); color:#fff; }
.step.done { color:#c9cdea; }
.step.done .num { background:rgba(74,222,128,.2); color:#4ade80; }
.step-line { width:26px; height:1px; background:rgba(255,255,255,.14); }

/* Section headings get a gradient accent bar */
.block-container h3 { position:relative; padding-left:.8rem; }
.block-container h3::before { content:""; position:absolute; left:0; top:.2em; bottom:.2em; width:4px; border-radius:4px;
    background:linear-gradient(180deg,#8b5cf6,#06b6d4); }
section[data-testid="stSidebar"] h3 { font-size:.8rem; text-transform:uppercase; letter-spacing:1.2px; color:#9ba3c7; padding-left:0; }
section[data-testid="stSidebar"] h3::before { display:none; }


/* ---- Tabs (current Streamlit uses role="tab"; old baseweb selectors no longer match) ---- */
[data-testid="stTabs"] [role="tablist"] {
    gap: 6px; background: rgba(255,255,255,.04); padding: 6px; border-radius: 14px;
    border: 1px solid rgba(255,255,255,.08); border-bottom: 1px solid rgba(255,255,255,.08) !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [role="tab"], [data-testid="stTab"] {
    padding: .55rem 1.15rem !important; border-radius: 10px !important; background: transparent;
    color: #9ba3c7 !important; font-weight: 600; border: none !important; border-bottom: none !important;
    box-shadow: none !important; height: auto !important; margin: 0 !important; white-space: nowrap;
}
[data-testid="stTabs"] [role="tab"]:hover { background: rgba(255,255,255,.06); color: #fff !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"], [data-testid="stTab"][data-selected] {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; color: #fff !important;
}
[data-testid="stTabs"] [role="tab"] *, [data-testid="stTab"] * { color: inherit !important; }
/* The red underline = Streamlit's SelectionIndicator element + a baseline ::after on the tab list */
.react-aria-SelectionIndicator, [data-testid="stTabs"] [class*="SelectionIndicator"] { display: none !important; }
[data-testid="stTabs"] [role="tablist"]::after, .react-aria-TabList::after { display: none !important; content: none !important; }
[data-testid="stTabs"] [role="tablist"] { overflow-y: hidden; }
/* Kill the default red/primary underline in every form Streamlit has used */
[data-testid="stTabs"] [role="tab"]::after, [data-testid="stTabs"] [role="tab"]::before,
[data-testid="stTab"]::after, [data-testid="stTab"]::before { display: none !important; content: none !important; }
[data-testid="stTabs"] [role="tablist"] > div:not([role="tab"]):not([data-testid="stTab"]),
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [role="tablist"] [style*="background-color"]:not([role="tab"]) { background: transparent !important; }

/* ---- Sidebar text never clips ---- */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; word-break: normal; }
section[data-testid="stSidebar"] ol, section[data-testid="stSidebar"] ul { padding-left: 1.2rem; }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-right: 1.2rem; }

/* Inputs */
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 12px !important; }
.stButton > button:focus-visible { outline: 2px solid #22d3ee; outline-offset: 2px; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'findings' not in st.session_state:
        st.session_state.findings = {}
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = GroqClient()
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'assumptions' not in st.session_state:
        st.session_state.assumptions = {}
    if 'assumptions_confirmed' not in st.session_state:
        st.session_state.assumptions_confirmed = False
    if 'sample_data_loaded' not in st.session_state:
        st.session_state.sample_data_loaded = False
    if 'assistant_open' not in st.session_state:
        st.session_state.assistant_open = True
    if 'assistant_theme' not in st.session_state:
        st.session_state.assistant_theme = 'dark'
    if 'assistant_pending' not in st.session_state:
        st.session_state.assistant_pending = None


def render_stepper():
    """Progress indicator: Upload > Review > Analyze > Story."""
    ss = st.session_state
    if ss.df is None:
        current = 0
    elif not ss.assumptions_confirmed:
        current = 1
    elif not ss.analysis_done:
        current = 2
    else:
        current = 3
    labels = ["Upload", "Review", "Analyze", "Story"]
    parts = []
    for i, label in enumerate(labels):
        cls = "done" if i < current else ("active" if i == current else "")
        num = "&#10003;" if i < current else str(i + 1)
        parts.append(f'<span class="step {cls}"><span class="num">{num}</span>{label}</span>')
        if i < len(labels) - 1:
            parts.append('<span class="step-line"></span>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div class="hero">
        <div class="hero-logo">{_logo_svg("hero", 64)}</div>
        <div class="hero-pill">AI-POWERED DATA STORYTELLING</div>
        <h1 class="main-header">DataStory AI</h1>
        <p class="sub-header">Upload a CSV and get an interactive, evidence-backed story with the exact code behind every insight.</p>
        <div class="hero-chips">
            <span class="chip">4-chapter story</span>
            <span class="chip">Show Calculation</span>
            <span class="chip">Spurious-correlation guardrails</span>
            <span class="chip">Works offline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_stepper()


def render_sidebar():
    """Render sidebar with configuration and info."""
    with st.sidebar:
        st.markdown("### Configuration")

        # Groq API status
        groq_client = st.session_state.groq_client
        if groq_client.is_configured():
            st.success(f"Groq AI Connected ({groq_client.config.provider})")
            st.caption(f"Model: {groq_client.config.model}")
        else:
            st.warning("Groq AI Not Configured")
            st.caption("Add GROQ_API_KEY to secrets or .env for AI explanations")
            st.caption("Works offline without API key - AI is optional!")

        st.divider()

        # Analysis options
        st.markdown("### Analysis Options")
        st.session_state.max_rows = st.number_input(
            "Max rows to analyze",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="Larger datasets will be sampled for speed"
        )

        st.session_state.min_correlation = st.slider(
            "Min correlation threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05
        )

        st.divider()

        # Export options
        st.markdown("### Export")
        if st.session_state.analysis_done:
            if st.button("Export Markdown", icon=":material/description:", use_container_width=True):
                md = export_to_markdown(
                    st.session_state.findings,
                    st.session_state.profile,
                    st.session_state.metadata,
                    st.session_state.df
                )
                st.download_button(
                    "Download .md",
                    md,
                    file_name=f"datastory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            if st.button("Export HTML", icon=":material/language:", use_container_width=True):
                html = export_to_html(
                    st.session_state.findings,
                    st.session_state.profile,
                    st.session_state.metadata,
                    st.session_state.df
                )
                st.download_button(
                    "Download .html",
                    html,
                    file_name=f"datastory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )

            if st.button("Export PDF", icon=":material/picture_as_pdf:", use_container_width=True):
                try:
                    pdf = export_to_pdf(
                        st.session_state.findings,
                        st.session_state.profile,
                        st.session_state.metadata,
                        st.session_state.df
                    )
                    st.download_button(
                        "Download .pdf",
                        pdf,
                        file_name=f"datastory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF export requires weasyprint: {e}")

        st.divider()

        # Info
        st.markdown("### About")
        st.markdown("""
        **DataStory AI** analyzes your CSV and creates a 4-chapter story:
        1. **Big Picture** - Overview & key metrics
        2. **Hidden Opportunities** - Trends & correlations
        3. **Unexpected Anomalies** - Outliers & quality issues
        4. **Recommended Actions** - What to do next

        **Key Feature:** Every finding has a "Show Calculation" expander
        with the exact Python code that computed it.

        Works 100% offline - AI narrative is optional!
        """)

        st.divider()
        st.caption("Built for Infinity AI Forge 2026 @ VIT-AP")


def render_upload_section():
    """Render CSV upload section with sample datasets."""
    st.markdown("### Upload Your Data")

    # Sample dataset buttons
    st.markdown("**Or try a sample dataset:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Tips Dataset", icon=":material/restaurant:", use_container_width=True, help="Restaurant tips data (244 rows)"):
            load_sample_dataset("tips")
            st.rerun()

    with col2:
        if st.button("Subscription Churn", icon=":material/trending_down:", use_container_width=True, help="Customer churn data (150 rows)"):
            load_sample_dataset("subscription_churn")
            st.rerun()

    with col3:
        if st.button("Guardrail Stress", icon=":material/shield:", use_container_width=True, help="Spurious correlation test (180 rows)"):
            load_sample_dataset("guardrail_stress")
            st.rerun()

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Drag and drop or click to browse. Supports various encodings and delimiters."
    )

    if uploaded_file is not None:
        with st.spinner("Loading and profiling data..."):
            try:
                # Load CSV
                df, metadata = load_csv_from_upload(uploaded_file)

                # Sample if too large
                if len(df) > st.session_state.max_rows:
                    df = df.sample(n=st.session_state.max_rows, random_state=42)
                    st.info(f"Dataset sampled to {st.session_state.max_rows:,} rows for analysis.")

                # Profile
                profile = profile_dataframe(df)

                # Store in session state
                st.session_state.df = df
                st.session_state.profile = profile
                st.session_state.metadata = metadata
                st.session_state.analysis_done = False
                st.session_state.findings = {}
                st.session_state.chat_history = []
                st.session_state.assumptions = {}
                st.session_state.assumptions_confirmed = False

                st.success(f"Loaded: {metadata['rows']:,} rows × {metadata['cols']} columns")
                st.caption(f"Encoding: {metadata['encoding']} | Delimiter: '{metadata['delimiter']}' | Header row: {metadata['header_row']}")

                if metadata['warnings']:
                    for w in metadata['warnings']:
                        st.warning(w)

            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.exception(e)


def load_sample_dataset(name: str):
    """Load a bundled sample dataset."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # Support both layouts: datastory/sample_data and ../sample_data
    sample_dir = next(
        (d for d in (os.path.join(here, 'sample_data'), os.path.join(here, '..', 'sample_data'))
         if os.path.isdir(d)),
        os.path.join(here, 'sample_data'),
    )

    dataset_files = {
        'tips': 'tips.csv',
        'subscription_churn': 'subscription_churn.csv',
        'guardrail_stress': 'guardrail_stress.csv',
    }

    if name not in dataset_files:
        st.error(f"Unknown sample dataset: {name}")
        return

    file_path = os.path.join(sample_dir, dataset_files[name])
    if not os.path.exists(file_path):
        st.error(f"Sample dataset not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        profile = profile_dataframe(df)

        st.session_state.df = df
        st.session_state.profile = profile
        st.session_state.metadata = {
            'rows': len(df),
            'cols': len(df.columns),
            'encoding': 'utf-8',
            'delimiter': ',',
            'header_row': 0,
            'warnings': []
        }
        st.session_state.analysis_done = False
        st.session_state.findings = {}
        st.session_state.chat_history = []
        st.session_state.assumptions = {}
        st.session_state.assumptions_confirmed = False
        st.session_state.sample_data_loaded = True

    except Exception as e:
        st.error(f"Error loading sample dataset: {e}")


def detect_assumptions(df: pd.DataFrame, profile: Dict) -> List[Dict]:
    """Detect assumptions about the data that user should confirm."""
    assumptions = []

    # Check for datetime columns
    for col in profile.get('datetime_columns', []):
        assumptions.append({
            'id': f'datetime_{col}',
            'type': 'datetime',
            'title': f'Column "{col}" treated as datetime',
            'description': f'This column was parsed as dates/times. Confirm this is correct.',
            'column': col,
            'default': True
        })

    # Check for ID columns
    for col in profile.get('id_columns', []):
        if col in df.columns and df[col].nunique() == len(df):
            assumptions.append({
                'id': f'id_{col}',
                'type': 'id',
                'title': f'Column "{col}" appears to be an ID',
                'description': f'All values are unique. This will be excluded from statistical analysis.',
                'column': col,
                'default': True
            })

    # Check for high cardinality categorical
    for col in profile.get('categorical_summary', {}):
        unique = profile['categorical_summary'][col]['unique']
        if unique > 50:
            assumptions.append({
                'id': f'high_card_{col}',
                'type': 'high_cardinality',
                'title': f'Column "{col}" has high cardinality ({unique} unique values)',
                'description': 'May be better treated as text/ID rather than categorical.',
                'column': col,
                'default': True
            })

    # Check for potential target column (if there's a binary column)
    for col in df.columns:
        if df[col].nunique() == 2 and profile.get('categorical_summary', {}).get(col):
            assumptions.append({
                'id': f'target_{col}',
                'type': 'target_heuristic',
                'title': f'Column "{col}" might be a target variable (binary)',
                'description': 'This binary column could be a label/target for analysis.',
                'column': col,
                'default': False
            })

    # Check row semantics
    if len(df) > 0:
        assumptions.append({
            'id': 'row_semantics',
            'type': 'row_semantics',
            'title': 'Each row represents one observation/transaction',
            'description': 'Analysis assumes rows are independent observations. Confirm this matches your data.',
            'column': None,
            'default': True
        })

    return assumptions


def render_assumption_checklist():
    """Render the assumption checklist before analysis."""
    if st.session_state.assumptions_confirmed:
        return True

    if not st.session_state.assumptions:
        df = st.session_state.df
        profile = st.session_state.profile
        # Store as dict keyed by id so each item's confirmed state is easy to look up
        st.session_state.assumptions = {a['id']: a for a in detect_assumptions(df, profile)}

    if not st.session_state.assumptions:
        st.session_state.assumptions_confirmed = True
        return True

    st.divider()
    st.markdown("### Assumption Checklist")
    st.markdown("Before generating the story, please review and confirm these detected assumptions:")

    all_confirmed = True
    for assumption_id, assumption in st.session_state.assumptions.items():
        confirmed = assumption.get('confirmed', assumption['default'])

        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            new_confirmed = st.checkbox("", value=confirmed, key=f"assume_{assumption_id}")
        with col2:
            css_class = "assumption-item assumption-confirmed" if new_confirmed else "assumption-item"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{assumption['title']}</strong><br>
                <small>{assumption['description']}</small>
            </div>
            """, unsafe_allow_html=True)

        assumption['confirmed'] = new_confirmed
        if not new_confirmed:
            all_confirmed = False

    if all_confirmed:
        if st.button("All Confirmed - Proceed to Analysis", icon=":material/check_circle:", type="primary", use_container_width=True):
            st.session_state.assumptions_confirmed = True
            st.rerun()
    else:
        st.warning("Please confirm all assumptions before proceeding.")

    return False


def render_data_preview():
    """Render data preview and profile summary."""
    if st.session_state.df is None:
        return

    df = st.session_state.df
    profile = st.session_state.profile

    st.divider()
    st.markdown("### Data Preview")

    n_rows, n_cols = df.shape
    numeric_cols = int(df.select_dtypes("number").shape[1])
    missing_pct = float(df.isna().sum().sum()) / max(n_rows * n_cols, 1) * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{n_rows:,}")
    k2.metric("Columns", f"{n_cols}")
    k3.metric("Numeric columns", f"{numeric_cols}")
    k4.metric("Missing cells", f"{missing_pct:.1f}%")

    # Preview tabs
    tab1, tab2, tab3 = st.tabs(["Sample Data", "Column Info", "Profile Summary"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 10 of {len(df):,} rows")

    with tab2:
        col_info = get_column_info(df)
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown(get_profile_summary_text(profile))

    # Show assumption checklist after preview
    if not st.session_state.assumptions_confirmed:
        render_assumption_checklist()


def _safe_detect(label: str, fn, *args) -> List[Dict]:
    """Run one detector. If it crashes, record a warning and carry on with the others."""
    try:
        result = fn(*args)
        # Downstream code expects a list of finding dicts; drop anything else.
        return [f for f in (result or []) if isinstance(f, dict)]
    except Exception as e:
        st.session_state.setdefault('analysis_warnings', []).append(
            f"{label} was skipped because of an error in its module: {type(e).__name__}: {e}")
        return []


def run_analysis():
    """Run all analysis modules and generate findings."""
    if st.session_state.df is None or st.session_state.profile is None:
        return

    if not st.session_state.assumptions_confirmed:
        st.warning("Please confirm assumptions before running analysis.")
        return

    df = st.session_state.df
    profile = st.session_state.profile

    with st.spinner("Analyzing data... This may take a moment."):
        findings = {}
        st.session_state.analysis_warnings = []

        # Run each detection module
        st.toast("Detecting trends...", icon=":material/trending_up:")
        findings['opportunities'] = _safe_detect('Trend detection', detect_trends, df, profile)

        st.toast("Mining correlations...", icon=":material/hub:")
        corr_findings = _safe_detect('Correlation detection', detect_correlations, df, profile)
        findings['opportunities'].extend(corr_findings)

        # Categorical-numeric relationships
        cat_num_findings = _safe_detect('Category/number relationships', detect_categorical_numeric_relationships, df, profile)
        findings['opportunities'].extend(cat_num_findings)

        st.toast("Finding anomalies...", icon=":material/report:")
        findings['anomalies'] = _safe_detect('Anomaly detection', detect_anomalies, df, profile)

        st.toast("Discovering segments...", icon=":material/category:")
        findings['segments'] = _safe_detect('Segment detection', detect_segments, df, profile)

        # Add segments to opportunities chapter
        findings['opportunities'].extend(findings['segments'])

        # Actions chapter - convert top findings to actions
        findings['actions'] = generate_action_recommendations(findings, profile)

        # Store findings
        st.session_state.findings = findings
        st.session_state.analysis_done = True

        # Generate AI explanations if Groq is configured
        if st.session_state.groq_client.is_configured():
            with st.spinner("Generating AI explanations..."):
                add_ai_explanations(findings, profile)

        st.toast("Analysis complete!", icon=":material/check_circle:")


def detect_categorical_numeric_relationships(df: pd.DataFrame, profile: Dict) -> List[Dict]:
    """Wrapper for categorical-numeric detection."""
    from datastory.analysis.correlations import detect_categorical_numeric_relationships
    return detect_categorical_numeric_relationships(df, profile)


def generate_action_recommendations(findings: Dict, profile: Dict) -> List[Dict]:
    """Generate actionable recommendations from findings."""
    actions = []

    # From correlations
    for f in findings.get('opportunities', []):
        if f['type'] == 'correlation' and abs(f['evidence'].get('correlation', 0)) > 0.6:
            ev = f['evidence']
            actions.append({
                'type': 'action',
                'title': f'Investigate {ev["column_1"]} → {ev["column_2"]} Relationship',
                'description': f'Strong {ev["direction"]} correlation (r={ev["correlation"]:.2f}) suggests '
                             f'{ev["column_1"]} may drive {ev["column_2"]}. Run controlled tests or '
                             f'check for confounding variables.',
                'evidence': {'source_finding': f['title'], 'correlation': ev['correlation']},
                'priority': 'high'
            })

    # From segments
    for f in findings.get('opportunities', []):
        if f['type'] in ['segment', 'segment_2d'] and abs(f['evidence'].get('z_score', 0)) > 2:
            ev = f['evidence']
            direction = 'high' if ev.get('z_score', 0) > 0 else 'low'
            actions.append({
                'type': 'action',
                'title': f'Target {direction.capitalize()}-Performing Segment',
                'description': f'Segment {f["title"]} shows {abs(ev.get("pct_difference", 0)):.0f}% {direction}er performance. '
                             f'Analyze what makes this segment different and replicate or investigate.',
                'evidence': {'source_finding': f['title'], 'segment': ev.get('segment_value', '')},
                'priority': 'medium'
            })

    # From anomalies
    for f in findings.get('anomalies', []):
        if f['type'] == 'outlier_iqr' and f['evidence'].get('outlier_percentage', 0) > 5:
            ev = f['evidence']
            actions.append({
                'type': 'action',
                'title': f'Review Outliers in {ev["column"]}',
                'description': f'{ev["outlier_count"]} outliers ({ev["outlier_percentage"]:.1f}%) detected. '
                             f'Verify data quality: are these errors, fraud, or genuine extreme events?',
                'evidence': {'source_finding': f['title'], 'outlier_count': ev['outlier_count']},
                'priority': 'high'
            })
        elif f['type'] == 'quality_issue' and f['evidence'].get('issue_type') == 'high_missingness':
            ev = f['evidence']
            actions.append({
                'type': 'action',
                'title': f'Address Missing Data in {ev["column"]}',
                'description': f'{ev["missing_percentage"]:.1f}% missing. Decide on imputation strategy '
                             f'(mean/median, model-based, or drop) based on missingness pattern.',
                'evidence': {'source_finding': f['title'], 'missing_pct': ev['missing_percentage']},
                'priority': 'high'
            })
        elif f['type'] == 'sudden_change':
            ev = f['evidence']
            actions.append({
                'type': 'action',
                'title': f'Investigate Sudden Change on {ev["date_after"]}',
                'description': f'{ev["percent_change"]:+.1f}% jump in {ev["numeric_column"]}. '
                             f'Check for external events, data pipeline issues, or policy changes.',
                'evidence': {'source_finding': f['title'], 'date': ev['date_after']},
                'priority': 'medium'
            })

    # Limit and sort by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    actions.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
    return actions[:5]


def add_ai_explanations(findings: Dict, profile: Dict):
    """Add AI explanations to all findings."""
    client = st.session_state.groq_client

    for chapter_id, chapter_findings in findings.items():
        for finding in chapter_findings:
            if 'ai_explanation' not in finding:
                try:
                    explanation = client.explain_finding(finding, profile)
                    finding['ai_explanation'] = explanation
                except Exception as e:
                    finding['ai_explanation'] = f"AI explanation failed: {e}"


def render_analyze_button():
    """Render the analyze button."""
    if st.session_state.df is not None and not st.session_state.analysis_done:
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Analyze & Generate Story", icon=":material/play_arrow:", type="primary", use_container_width=True):
                run_analysis()
                st.rerun()


def render_story():
    """Render the 4-chapter data story."""
    if not st.session_state.analysis_done:
        return

    st.divider()
    for msg in st.session_state.get('analysis_warnings', []):
        st.warning(msg)

    st.markdown("## Your Data Story")

    # Executive summary
    exec_summary = generate_executive_summary(st.session_state.findings, st.session_state.profile)
    with st.expander("Executive Summary", expanded=True):
        st.markdown(exec_summary)

    # Chapter tabs
    render_chapter_tabs(st.session_state.findings, st.session_state.profile, st.session_state.df)

    
    


CHAPTER_TITLES = {
    'opportunities': 'Hidden Opportunities',
    'anomalies': 'Unexpected Anomalies',
    'actions': 'Recommended Actions',
}


def _pdf_text(text) -> str:
    """Make text safe for the PDF's built-in fonts: no emoji, no unsupported glyphs."""
    text = strip_emoji(str(text))
    for old, new in (("\u2192", "->"), ("\u2265", ">="), ("\u2264", "<="), ("\u2248", "~"),
                     ("\u2212", "-"), ("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')):
        text = text.replace(old, new)
    return text.encode("cp1252", "ignore").decode("cp1252")


def _md_to_pdf_markup(text) -> str:
    """Escape text for reportlab and convert **bold** / *italic* markdown."""
    safe = html.escape(_pdf_text(text), quote=False)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", safe)
    return safe.replace("`", "")


def build_summary_pdf(summary_md: str, findings: Dict, metadata: Optional[Dict] = None) -> bytes:
    """Build a clean, branded PDF of the executive summary and every chapter's findings."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

    violet, navy, ink, muted = colors.HexColor("#6d28d9"), colors.HexColor("#1e3a8a"), colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
    st_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=violet, spaceAfter=2)
    st_meta = ParagraphStyle("m", fontName="Helvetica", fontSize=9, leading=12, textColor=muted, spaceAfter=6)
    st_h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=navy, spaceBefore=14, spaceAfter=4)
    st_h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=ink, spaceBefore=8, spaceAfter=2)
    st_body = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14.5, textColor=ink, alignment=TA_LEFT, spaceAfter=4)
    st_bullet = ParagraphStyle("bl", parent=st_body, leftIndent=12, bulletIndent=2, spaceAfter=2)
    st_ai = ParagraphStyle("ai", parent=st_body, fontName="Helvetica-Oblique", textColor=colors.HexColor("#4b5563"),
                           leftIndent=8, borderPadding=(2, 2, 2, 6), spaceAfter=6)

    story = [Paragraph("DataStory AI", st_title)]
    meta_bits = [f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}"]
    if metadata and metadata.get("rows") is not None:
        meta_bits.insert(0, f"{metadata.get('rows', 0):,} rows x {metadata.get('cols', 0)} columns")
    story += [Paragraph(" | ".join(meta_bits), st_meta),
              HRFlowable(width="100%", thickness=2, color=violet, spaceAfter=6)]

    # Executive summary (markdown text from generate_executive_summary)
    story.append(Paragraph("Executive Summary", st_h1))
    for raw in _pdf_text(summary_md).split("\n"):
        line = raw.strip()
        if not line or set(line) <= set("-=_*"):
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading.lower().strip(" *:") == "executive summary":
                continue  # already shown as the section title above
            story.append(Paragraph(_md_to_pdf_markup(heading), st_h2))
        elif line[:2] in ("- ", "* ") or line.startswith("\u2022"):
            story.append(Paragraph(_md_to_pdf_markup(line.lstrip("-*\u2022 ").strip()), st_bullet, bulletText="\u2022"))
        else:
            story.append(Paragraph(_md_to_pdf_markup(line), st_body))

    # Chapters ('segments' are already merged into 'opportunities', so skip them to avoid duplicates)
    for chapter_id, items in findings.items():
        if chapter_id == "segments" or not items:
            continue
        story.append(Paragraph(CHAPTER_TITLES.get(chapter_id, chapter_id.replace("_", " ").title()), st_h1))
        story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#d1d5db"), spaceAfter=4))
        for f in items:
            block = [Paragraph(_md_to_pdf_markup(f.get("title", "")), st_h2),
                     Paragraph(_md_to_pdf_markup(f.get("description", "")), st_body)]
            if f.get("ai_explanation"):
                block.append(Paragraph("AI insight: " + _md_to_pdf_markup(f["ai_explanation"]), st_ai))
            story.append(KeepTogether(block))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 10 * mm, "DataStory AI - evidence-backed data story")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.setFillColor(violet)
        canvas.rect(0, A4[1] - 3 * mm, A4[0], 3 * mm, stroke=0, fill=1)
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title="DataStory AI Summary")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def render_footer():
    """Render footer with export options."""
    if not st.session_state.analysis_done:
        return

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        # Export findings as JSON
        if st.button("Export Findings (JSON)", icon=":material/download:"):
            export_data = {
                'profile': st.session_state.profile,
                'findings': st.session_state.findings,
                'metadata': st.session_state.metadata
            }
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="datastory_findings.json",
                mime="application/json"
            )

    with col2:
        # Export executive summary as PDF
        if st.button("Export Summary (PDF)", icon=":material/picture_as_pdf:"):
            try:
                summary = generate_executive_summary(st.session_state.findings, st.session_state.profile)
                st.session_state.summary_pdf = build_summary_pdf(
                    summary, st.session_state.findings, st.session_state.get('metadata'))
            except ImportError:
                st.session_state.summary_pdf = None
                st.error("PDF export needs the reportlab package. Run: pip install reportlab")
            except Exception as e:
                st.session_state.summary_pdf = None
                st.error(f"Could not build the PDF: {e}")
        if st.session_state.get('summary_pdf'):
            st.download_button(
                label="Download PDF",
                data=st.session_state.summary_pdf,
                file_name=f"datastory_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                icon=":material/download:",
            )

    with col3:
        # Reset button
        if st.button("Analyze New File", icon=":material/refresh:"):
            for key in ['df', 'profile', 'findings', 'analysis_done', 'metadata', 'chat_history', 'summary_pdf']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()





# ---------------------------------------------------------------------------
# Floating support assistant (bottom-right chat widget)
# ---------------------------------------------------------------------------
# Needs Streamlit >= 1.39 (st.container(key=...)) and Material icon labels.
ASSISTANT_NAME = "DataStory Assistant"
ASSISTANT_SUBTITLE = "AI data support"

QUICK_ACTIONS = [
    (":material/trending_up:", "Explain the trends", "What is the general trend of the data?"),
    (":material/report:", "Find anomalies", "Are there any major anomalies?"),
    (":material/hub:", "Key correlations", "What are the key correlations?"),
    (":material/summarize:", "Summarize findings", "Give me a summary of findings."),
]


def _logo_svg(uid: str, size: int = 40) -> str:
    """Brand mark (bar-chart glyph on a gradient disc). Inline SVG, no emoji."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="g{uid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#8b5cf6"/><stop offset="1" stop-color="#06b6d4"/></linearGradient></defs>'
        f'<circle cx="24" cy="24" r="24" fill="url(#g{uid})"/>'
        f'<rect x="12.5" y="26" width="5.5" height="9" rx="1.8" fill="#fff" opacity=".85"/>'
        f'<rect x="21.25" y="19" width="5.5" height="16" rx="1.8" fill="#fff"/>'
        f'<rect x="30" y="12" width="5.5" height="23" rx="1.8" fill="#fff" opacity=".92"/></svg>'
    )


def _assistant_css(theme: str) -> str:
    """CSS that pins the assistant panel / launcher to the bottom-right corner."""
    if theme == "light":
        v = dict(bg="rgba(255,255,255,.97)", text="#111827", muted="#6b7280",
                 card="#f1f3f9", border="#e2e5ef", bot="#f1f3f9", input_bg="#ffffff")
    else:
        v = dict(bg="rgba(14,17,32,.94)", text="#e8eaf6", muted="#9ba3c7",
                 card="rgba(255,255,255,.06)", border="rgba(255,255,255,.12)",
                 bot="rgba(255,255,255,.07)", input_bg="rgba(255,255,255,.04)")
    return f"""
<style>
.st-key-diva_panel {{
    --dv-bg:{v['bg']}; --dv-text:{v['text']}; --dv-muted:{v['muted']};
    --dv-card:{v['card']}; --dv-border:{v['border']}; --dv-bot:{v['bot']};
    position: fixed; right: 24px; bottom: 24px; z-index: 999999;
    width: min(400px, calc(100vw - 32px));
    background: var(--dv-bg); color: var(--dv-text);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--dv-border); border-radius: 20px;
    box-shadow: 0 24px 60px rgba(0,0,0,.5), 0 0 0 1px rgba(139,92,246,.18);
    overflow: hidden; gap: 0 !important; padding: 0 !important;
    animation: dv-pop .28s ease-out;
}}
@keyframes dv-pop {{ from {{ opacity:0; transform: translateY(14px) scale(.98); }} to {{ opacity:1; transform:none; }} }}
.st-key-diva_panel p, .st-key-diva_panel span, .st-key-diva_panel label {{ color: inherit; }}

/* Header */
.st-key-diva_header {{
    background: linear-gradient(110deg, #4c1d95 0%, #1e3a8a 55%, #0e7490 100%);
    padding: 12px 14px; gap: 0 !important;
}}
.st-key-diva_header [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; align-items: center; gap: .15rem; }}
.st-key-diva_header [data-testid="stColumn"] {{ min-width: 0 !important; }}
.st-key-diva_header button, .st-key-diva_header [data-testid="stPopover"] button {{
    background: transparent !important; border: none !important; color: #fff !important;
    padding: .3rem .4rem; min-height: 0; box-shadow: none !important; transform: none !important;
}}
.st-key-diva_header button:hover {{ background: rgba(255,255,255,.16) !important; transform: none !important; }}
.dv-head {{ display:flex; align-items:center; gap:10px; }}
.dv-logo {{ position:relative; width:40px; height:40px; flex:none; }}
.dv-logo::after {{ content:""; position:absolute; right:-1px; bottom:0; width:10px; height:10px;
    background:#22c55e; border:2px solid #1e3a8a; border-radius:50%; }}
.dv-name {{ color:#fff; font-weight:700; font-size:.98rem; line-height:1.15; }}
.dv-sub {{ color:rgba(255,255,255,.72); font-size:.72rem; }}

/* Body */
.st-key-diva_body {{ padding: 0 16px; }}
.dv-welcome {{ text-align:center; padding-top:16px; }}
.dv-big-logo {{ width:84px; height:84px; margin:0 auto 12px; padding:6px; border-radius:50%;
    background: radial-gradient(circle, rgba(139,92,246,.35), transparent 70%); }}
.dv-welcome h3 {{ margin:.2rem 0 .4rem; font-size:1.25rem; font-weight:700; color:var(--dv-text); letter-spacing:-.3px; }}
.dv-welcome p {{ color:var(--dv-muted); font-size:.85rem; line-height:1.5; margin-bottom:.9rem; }}

.st-key-diva_qa {{ gap:.5rem !important; }}
.st-key-diva_qa button {{
    width:100%; display:flex; justify-content:flex-start; text-align:left;
    background:var(--dv-card) !important; color:var(--dv-text) !important;
    border:1px solid var(--dv-border) !important; border-radius:12px;
    padding:.7rem 1rem; font-weight:600; transform:none !important; box-shadow:none !important;
}}
.st-key-diva_qa button > div {{ width:100%; justify-content:flex-start; }}
.st-key-diva_qa button::after {{ content:"\\203A"; margin-left:auto; color:var(--dv-muted); font-size:1.25rem; line-height:1; }}
.st-key-diva_qa button:hover {{
    background:rgba(139,92,246,.16) !important; border-color:#8b5cf6 !important;
    color:var(--dv-text) !important; transform:none !important; box-shadow:none !important;
}}

.dv-row {{ display:flex; margin:8px 0; }}
.dv-row.user {{ justify-content:flex-end; }}
.dv-bubble {{ max-width:86%; padding:.6rem .85rem; border-radius:16px; font-size:.88rem; line-height:1.5; word-wrap:break-word; }}
.dv-row.user .dv-bubble {{ background:linear-gradient(120deg,#7c3aed,#2563eb); color:#fff; border-bottom-right-radius:5px; }}
.dv-row.bot .dv-bubble {{ background:var(--dv-bot); color:var(--dv-text); border:1px solid var(--dv-border); border-bottom-left-radius:5px; }}

/* Input row */
.st-key-diva_input {{ padding: 10px 16px 16px; }}
.st-key-diva_input [data-testid="stChatInput"] {{
    background:{v['input_bg']} !important; border:1.5px solid rgba(139,92,246,.7) !important;
    border-radius:999px !important; box-shadow:none !important; overflow:hidden;
}}
.st-key-diva_input [data-testid="stChatInput"]:focus-within {{
    border-color:#22d3ee !important; box-shadow:0 0 0 3px rgba(34,211,238,.18) !important;
}}
.st-key-diva_input [data-testid="stChatInput"] > div,
.st-key-diva_input [data-testid="stChatInput"] div[data-baseweb],
.st-key-diva_input [data-testid="stChatInputTextArea"] {{
    background:transparent !important; border:none !important; box-shadow:none !important; border-radius:999px !important;
}}
.st-key-diva_input [data-testid="stChatInput"] textarea {{ color:var(--dv-text) !important; background:transparent !important; padding-left:.9rem; }}
.st-key-diva_input [data-testid="stChatInput"] textarea::placeholder {{ color:var(--dv-muted) !important; opacity:1; }}
.st-key-diva_input [data-testid="stChatInputSubmitButton"] {{
    background:linear-gradient(120deg,#8b5cf6,#06b6d4) !important; color:#fff !important; border-radius:50% !important;
}}

/* Launcher (shown when the panel is minimised) */
.st-key-diva_fab {{ position:fixed; right:24px; bottom:24px; z-index:999999; width:auto; }}
.st-key-diva_fab button {{
    width:60px; height:60px; border-radius:50% !important; padding:0;
    background:linear-gradient(120deg,#8b5cf6,#06b6d4) !important; color:#fff !important;
    border:none !important; box-shadow:0 10px 30px rgba(139,92,246,.55);
}}
.st-key-diva_fab button:hover {{ transform: scale(1.07) !important; }}

@media (max-width: 640px) {{
    .st-key-diva_panel {{ right:16px; bottom:16px; }}
    .st-key-diva_fab {{ right:16px; bottom:16px; }}
}}
</style>
"""


def _format_bubble(text: str) -> str:
    """Escape text, then allow **bold** and line breaks."""
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe.replace("\n", "<br>")


def _assistant_context(max_findings: int = 14) -> str:
    """Compact description of the dataset + findings for the LLM / offline answers."""
    df = st.session_state.get("df")
    if df is None:
        return "No dataset loaded."
    lines = [
        f"Dataset: {len(df):,} rows x {len(df.columns)} columns.",
        "Columns: " + ", ".join(f"{c} ({df[c].dtype})" for c in list(df.columns)[:40]),
    ]
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        lines.append("Missing values: " + ", ".join(f"{c}={n}" for c, n in missing.items()))
    count = 0
    for chapter, items in (st.session_state.get("findings") or {}).items():
        for f in items:
            if count >= max_findings:
                break
            lines.append(f"[{chapter}/{f.get('type', '')}] {f.get('title', '')}: {f.get('description', '')}")
            count += 1
    return "\n".join(lines)


def _ask_llm(question: str) -> Optional[str]:
    """Ask the configured Groq model. Returns None if unavailable or on any error."""
    client = st.session_state.get("groq_client")
    if client is None or not client.is_configured():
        return None
    try:
        import os
        import requests

        cfg = client.config
        api_key = getattr(cfg, "api_key", None) or os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None
        if not api_key:
            return None
        base_url = (getattr(cfg, "base_url", None) or "https://api.groq.com/openai/v1").rstrip("/")

        messages = [{
            "role": "system",
            "content": (
                f"You are {ASSISTANT_NAME}, a friendly support assistant inside a data-analysis app. "
                "Answer concisely (under 120 words) using ONLY the dataset context below. "
                "If the context does not contain the answer, say so and suggest what to run in the app. "
                "Do not use emojis.\n\n"
                + _assistant_context()
            ),
        }]
        for m in st.session_state.get("chat_history", [])[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})

        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": cfg.model, "messages": messages, "temperature": 0.3, "max_tokens": 400},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _offline_answer(question: str) -> str:
    """Rule-based answer built from the computed findings (no API key needed)."""
    df = st.session_state.get("df")
    if df is None:
        return "Upload a CSV (or pick a sample dataset) first, then ask me anything about it."

    findings = st.session_state.get("findings") or {}
    opps = findings.get("opportunities", [])
    anomalies = findings.get("anomalies", [])
    actions = findings.get("actions", [])
    q = question.lower()

    def bullets(items: List[Dict], n: int = 3) -> str:
        return "\n".join(f"• **{f.get('title', '')}** — {f.get('description', '')}" for f in items[:n])

    if any(k in q for k in ("row", "column", "shape", "size", "how big")):
        return f"Your data has **{len(df):,} rows** and **{len(df.columns)} columns**: " + ", ".join(map(str, df.columns[:12])) + ("..." if len(df.columns) > 12 else "")

    if not st.session_state.get("analysis_done"):
        return ("I haven't analysed this dataset yet. Click **Analyze & Generate Story** "
                "and I'll be able to talk about trends, correlations and anomalies.")

    if any(k in q for k in ("anomal", "outlier", "unusual", "quality", "missing")):
        return ("Here's what stood out:\n" + bullets(anomalies)) if anomalies else "I didn't find any major anomalies."
    if "trend" in q:
        items = [f for f in opps if "trend" in str(f.get("type", "")) or "change" in str(f.get("type", ""))]
        return ("Trends I found:\n" + bullets(items)) if items else "No clear time-based trends were detected in this data."
    if "correl" in q or "relationship" in q:
        items = [f for f in opps if f.get("type") == "correlation"]
        return ("Strongest relationships:\n" + bullets(items)) if items else "No correlations passed the current threshold."
    if "segment" in q or "group" in q:
        items = [f for f in opps if "segment" in str(f.get("type", ""))]
        return ("Notable segments:\n" + bullets(items)) if items else "No standout segments were found."
    if any(k in q for k in ("action", "recommend", "next", "should")):
        return ("Suggested next steps:\n" + bullets(actions)) if actions else "No specific actions were generated for this data."
    if any(k in q for k in ("summary", "summar", "finding", "overview", "key")):
        parts = [bullets(opps, 2), bullets(anomalies, 2), bullets(actions, 1)]
        text = "\n".join(p for p in parts if p)
        return ("Quick summary:\n" + text) if text else "No findings yet."
    return ("I can help with **trends**, **correlations**, **anomalies**, **segments**, "
            "**recommended actions** or a **summary**. Which one would you like?")


def generate_assistant_reply(question: str) -> str:
    """LLM answer when Groq is configured, otherwise a findings-based answer."""
    return _ask_llm(question) or _offline_answer(question)


def _queue_question(q: str):
    st.session_state.assistant_pending = q


def render_assistant():
    """Render the floating, bottom-right support assistant."""
    ss = st.session_state
    st.markdown(_assistant_css(ss.assistant_theme), unsafe_allow_html=True)

    # Minimised: only the round launcher button is shown
    if not ss.assistant_open:
        with st.container(key="diva_fab"):
            if st.button(":material/chat:", key="diva_open", help=f"Chat with {ASSISTANT_NAME}"):
                ss.assistant_open = True
                st.rerun()
        return

    with st.container(key="diva_panel"):
        # ---- Header -------------------------------------------------------
        with st.container(key="diva_header"):
            c_title, c_theme, c_min, c_menu = st.columns([0.58, 0.14, 0.14, 0.14])
            with c_title:
                st.markdown(
                    f'<div class="dv-head"><div class="dv-logo">{_logo_svg("h", 40)}</div>'
                    f'<div><div class="dv-name">{html.escape(ASSISTANT_NAME)}</div>'
                    f'<div class="dv-sub">{html.escape(ASSISTANT_SUBTITLE)}</div></div></div>',
                    unsafe_allow_html=True,
                )
            with c_theme:
                icon = ":material/dark_mode:" if ss.assistant_theme == "light" else ":material/light_mode:"
                if st.button(icon, key="diva_theme", help="Toggle theme"):
                    ss.assistant_theme = "dark" if ss.assistant_theme == "light" else "light"
                    st.rerun()
            with c_min:
                if st.button(":material/remove:", key="diva_min", help="Minimise"):
                    ss.assistant_open = False
                    st.rerun()
            with c_menu:
                with st.popover(":material/more_vert:"):
                    if st.button("Clear chat", key="diva_clear", icon=":material/delete:", use_container_width=True):
                        ss.chat_history = []
                        st.rerun()

        # ---- Answer a queued question (quick action or typed) ----------------
        pending = ss.assistant_pending
        if pending:
            ss.assistant_pending = None
            ss.chat_history.append({"role": "user", "content": pending})
            with st.spinner("Thinking..."):
                reply = generate_assistant_reply(pending)
            ss.chat_history.append({"role": "assistant", "content": reply})

        # ---- Body: welcome screen or conversation --------------------------
        with st.container(height=380, border=False, key="diva_body"):
            if not ss.chat_history:
                intro = (
                    f"Hi, I'm {html.escape(ASSISTANT_NAME)}. I can explain trends, correlations, "
                    "anomalies and recommendations from your data. What would you like to know?"
                    if ss.df is not None else
                    f"Hi, I'm {html.escape(ASSISTANT_NAME)}. Upload a CSV or load a sample dataset "
                    "and I'll help you understand it."
                )
                st.markdown(
                    f'<div class="dv-welcome"><div class="dv-big-logo">{_logo_svg("w", 72)}</div>'
                    f'<h3>How can I help you today?</h3><p>{intro}</p></div>',
                    unsafe_allow_html=True,
                )
                with st.container(key="diva_qa"):
                    for i, (icon, label, question) in enumerate(QUICK_ACTIONS):
                        st.button(f"{icon} {label}", key=f"diva_qa_{i}",
                                  on_click=_queue_question, args=(question,))
            else:
                for m in ss.chat_history:
                    role = "user" if m["role"] == "user" else "bot"
                    st.markdown(
                        f'<div class="dv-row {role}"><div class="dv-bubble">{_format_bubble(m["content"])}</div></div>',
                        unsafe_allow_html=True,
                    )

        # ---- Input ----------------------------------------------------------
        with st.container(key="diva_input"):
            typed = st.chat_input("Type your message...", key="diva_chat_input")
            if typed:
                ss.assistant_pending = typed
                st.rerun()


def render_main_content():
    if st.session_state.df is None:
        render_upload_section()
    else:
        render_data_preview()
        render_analyze_button()
        render_story()
        render_footer()


def main():
    initialize_session_state()
    render_header()
    render_sidebar()
    render_main_content()
    render_assistant()


if __name__ == "__main__":
    main()