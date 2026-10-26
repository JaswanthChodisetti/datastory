"""Evidence formatting module - formats evidence for display in the UI."""

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, List


def format_finding_evidence(evidence: Dict[str, Any]):
    """Format and display finding evidence in a readable way."""
    if not evidence:
        st.caption("No evidence available")
        return

    # Create a clean display of key evidence
    cols = st.columns(2)
    items = list(evidence.items())
    mid = len(items) // 2 + len(items) % 2

    for i, (key, value) in enumerate(items):
        # Skip internal fields
        if key in ['row_indices', 'example_row_indices', 'group_means', 'segment_values']:
            continue

        col = cols[i % 2]
        with col:
            # Format key nicely
            display_key = key.replace('_', ' ').title()
            # Format value
            if isinstance(value, float):
                display_value = f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
            elif isinstance(value, (int,)):
                display_value = f"{value:,}"
            elif isinstance(value, dict):
                display_value = json.dumps(value, indent=2)
            elif isinstance(value, list):
                display_value = f"[{len(value)} items]" if len(value) > 5 else ", ".join(str(v) for v in value)
            else:
                display_value = str(value)

            col.metric(display_key, display_value)


def format_evidence_table(evidence: Dict[str, Any]):
    """Format evidence as a table for detailed view."""
    if not evidence:
        st.info("No evidence data available")
        return

    # Convert to DataFrame for display
    rows = []
    for key, value in evidence.items():
        if key in ['row_indices', 'example_row_indices']:
            rows.append({'Field': key.replace('_', ' ').title(), 'Value': f"{len(value)} row indices", 'Type': 'List'})
        elif isinstance(value, dict):
            rows.append({'Field': key.replace('_', ' ').title(), 'Value': json.dumps(value, indent=2), 'Type': 'Dict'})
        elif isinstance(value, list):
            if len(value) <= 10:
                rows.append({'Field': key.replace('_', ' ').title(), 'Value': ", ".join(str(v) for v in value), 'Type': 'List'})
            else:
                rows.append({'Field': key.replace('_', ' ').title(), 'Value': f"{len(value)} items (showing first 10): {', '.join(str(v) for v in value[:10])}...", 'Type': 'List'})
        else:
            rows.append({'Field': key.replace('_', ' ').title(), 'Value': str(value), 'Type': type(value).__name__})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def format_row_evidence(df: pd.DataFrame, row_indices: List[int], columns: List[str] = None) -> pd.DataFrame:
    """Format specific rows as evidence table."""
    if not row_indices:
        return pd.DataFrame()

    # Get rows
    evidence_df = df.loc[row_indices].copy()

    # Limit columns if specified
    if columns:
        available_cols = [c for c in columns if c in evidence_df.columns]
        if available_cols:
            evidence_df = evidence_df[available_cols]

    # Limit rows displayed
    if len(evidence_df) > 20:
        evidence_df = evidence_df.head(20)
        st.caption(f"Showing first 20 of {len(row_indices)} rows")

    return evidence_df


def create_evidence_download(evidence: Dict[str, Any], finding_title: str) -> str:
    """Create JSON string for evidence download."""
    export_data = {
        'finding': finding_title,
        'evidence': evidence,
        'exported_at': pd.Timestamp.now().isoformat()
    }
    return json.dumps(export_data, indent=2, default=str)


def render_evidence_expander(evidence: Dict[str, Any], title: str = "Evidence Details"):
    """Render evidence in an expander with download option."""
    with st.expander(title, expanded=False):
        format_evidence_table(evidence)

        # Download button
        json_str = create_evidence_download(evidence, title)
        st.download_button(
            label="📥 Download Evidence (JSON)",
            data=json_str,
            file_name=f"evidence_{title.lower().replace(' ', '_')}.json",
            mime="application/json"
        )