"""Support response logic for TaleMate AI.
Handles rule-based fallbacks and specific product guidance.
"""

from typing import Dict, Any, List

SUPPORT_GUIDANCE = {
    "upload": (
        "To upload your data, use the 'Upload Your Data' section on the main page. "
        "We support CSV files. For best results, ensure your file has a clear header row "
        "and consistent delimiters. If you're just testing, try one of our sample datasets!"
    ),
    "insight": (
        "Insights are detected by our 'Sensor Suite'—a set of deterministic statistical tests. "
        "A trend is flagged if there's a significant change over time, and an anomaly is "
        "flagged if a value is more than 3 standard deviations from the mean. "
        "You can see the exact math by clicking 'Show Calculation' on any finding."
    ),
    "evidence": (
        "Every finding in DataTale AI is evidence-backed. The 'Evidence Explorer' maps "
        "each insight to the exact rows in your CSV that prove it. This ensures zero "
        "hallucinations—the AI only narrates facts already calculated by the sensors."
    ),
    "export": (
        "You can export your curated data story as Markdown, HTML, or PDF via the 'Export' "
        "section in the sidebar. The HTML export is recommended as it preserves "
        "interactive Plotly charts and a professional layout."
    ),
    "default": (
        "I'm not entirely sure about that, but I can help you with data uploads, "
        "explaining insights, verifying evidence, or exporting your report. "
        "Could you try rephrasing your question?"
    )
}

def get_support_response(query: str, session_state: Any) -> str:
    """
    Provide a targeted response for support queries.
    If the query matches a known support topic, return the guidance.
    Otherwise, attempt to use the LLM if configured.
    """
    query_lower = query.lower()

    if "upload" in query_lower or "csv" in query_lower or "file" in query_lower:
        return SUPPORT_GUIDANCE["upload"]
    if "insight" in query_lower or "how" in query_lower and "detect" in query_lower:
        return SUPPORT_GUIDANCE["insight"]
    if "evidence" in query_lower or "verify" in query_lower or "hallucination" in query_lower:
        return SUPPORT_GUIDANCE["evidence"]
    if "export" in query_lower or "download" in query_lower or "report" in query_lower:
        return SUPPORT_GUIDANCE["export"]

    # Fallback to LLM if available
    if hasattr(session_state, 'groq_client') and session_state.groq_client.is_configured():
        # Construct a support-specific context for the LLM
        context = {
            "question": query,
            "columns": session_state.df.columns.tolist() if session_state.df is not None else [],
            "profile": session_state.profile if session_state.profile else {},
            "findings_summary": "User is asking a support question about the application."
        }
        return session_state.groq_client.explain_followup(context)

    return SUPPORT_GUIDANCE["default"]
