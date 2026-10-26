"""Fallback summaries when AI explanations are unavailable."""

from typing import Dict, Any, List
import pandas as pd


def generate_finding_summary(finding: Dict[str, Any]) -> str:
    """Generate a template-based summary for a finding."""
    finding_type = finding.get('type', 'finding')
    evidence = finding.get('evidence', {})

    generators = {
        'trend': _summarize_trend,
        'sequential_trend': _summarize_sequential_trend,
        'correlation': _summarize_correlation,
        'categorical_numeric': _summarize_categorical_numeric,
        'outlier_iqr': _summarize_outlier,
        'outlier_zscore': _summarize_outlier,
        'sudden_change': _summarize_sudden_change,
        'segment': _summarize_segment,
        'segment_2d': _summarize_segment_2d,
        'quality_issue': _summarize_quality_issue,
    }

    generator = generators.get(finding_type, _summarize_generic)
    return generator(evidence)


def _summarize_trend(ev: Dict) -> str:
    direction = ev.get('trend_direction', 'upward')
    pct = ev.get('change_pct', 0)
    col = ev.get('numeric_column', 'metric')
    dt = ev.get('datetime_column', 'time')
    p_val = ev.get('mann_kendall_p', 1)
    return (f"{col} shows a statistically significant {direction} trend over {dt} "
            f"(Mann-Kendall p={p_val:.4f}), changing by {pct:+.1f}% overall.")


def _summarize_sequential_trend(ev: Dict) -> str:
    direction = ev.get('trend_direction', 'upward')
    tau = ev.get('mann_kendall_tau', 0)
    p_val = ev.get('mann_kendall_p', 1)
    col = ev.get('numeric_column', 'metric')
    return (f"{col} shows a {direction} sequential trend in row order "
            f"(Kendall τ={tau:.3f}, p={p_val:.4f}).")


def _summarize_correlation(ev: Dict) -> str:
    c1 = ev.get('column_1', 'X')
    c2 = ev.get('column_2', 'Y')
    corr = ev.get('correlation', 0)
    method = ev.get('method', 'pearson')
    p_val = ev.get('p_value', 1)
    direction = 'positive' if corr > 0 else 'negative'
    return (f"{c1} and {c2} have a {direction} {method} correlation "
            f"(r={corr:.3f}, p={p_val:.4f}, n={ev.get('sample_size', 0)}).")


def _summarize_categorical_numeric(ev: Dict) -> str:
    num_col = ev.get('numeric_column', 'metric')
    cat_col = ev.get('categorical_column', 'category')
    p_val = ev.get('p_value', 1)
    max_g = ev.get('max_group', '')
    min_g = ev.get('min_group', '')
    diff = ev.get('difference_pct', 0)
    return (f"{num_col} differs significantly across {cat_col} (Kruskal-Wallis p={p_val:.4f}). "
            f"{max_g} is {diff:.1f}% higher than {min_g}.")


def _summarize_outlier(ev: Dict) -> str:
    col = ev.get('column', 'metric')
    count = ev.get('outlier_count', 0)
    pct = ev.get('outlier_percentage', 0)
    method = ev.get('method', 'IQR')
    return (f"{count} outliers ({pct:.1f}%) detected in {col} using {method} method. "
            f"Normal range: [{ev.get('lower_bound', 0):.2f}, {ev.get('upper_bound', 0):.2f}].")


def _summarize_sudden_change(ev: Dict) -> str:
    col = ev.get('numeric_column', 'metric')
    pct = ev.get('percent_change', 0)
    direction = ev.get('direction', 'change')
    date = ev.get('date_after', 'a point')
    return (f"Sudden {direction} of {pct:.1f}% in {col} at {date} "
            f"({ev.get('value_before', 0):.2f} → {ev.get('value_after', 0):.2f}).")


def _summarize_segment(ev: Dict) -> str:
    num_col = ev.get('numeric_column', 'metric')
    cat_col = ev.get('categorical_column', 'category')
    seg_val = ev.get('segment_value', 'segment')
    pct = ev.get('pct_difference', 0)
    count = ev.get('segment_count', 0)
    direction = 'higher' if pct > 0 else 'lower'
    return (f"Segment {cat_col}={seg_val} has {abs(pct):.1f}% {direction} {num_col} "
            f"than overall average (n={count}, z={ev.get('z_score', 0):.2f}).")


def _summarize_segment_2d(ev: Dict) -> str:
    num_col = ev.get('numeric_column', 'metric')
    cats = ev.get('categorical_columns', [])
    vals = ev.get('segment_values', {})
    pct = ev.get('pct_difference', 0)
    count = ev.get('segment_count', 0)
    direction = 'highest' if pct > 0 else 'lowest'
    cat_str = " × ".join(f"{c}={vals.get(c, '')}" for c in cats)
    return (f"Combination {cat_str} has the {direction} {num_col} "
            f"({ev.get('segment_mean', 0):.2f} vs {ev.get('overall_mean', 0):.2f} overall, n={count}).")


def _summarize_quality_issue(ev: Dict) -> str:
    issue = ev.get('issue_type', 'issue')
    if issue == 'high_missingness':
        col = ev.get('column', 'column')
        pct = ev.get('missing_percentage', 0)
        return f"{col} has {pct:.1f}% missing values — consider imputation or removal."
    elif issue == 'duplicates':
        count = ev.get('duplicate_count', 0)
        pct = ev.get('duplicate_percentage', 0)
        return f"{count} duplicate rows ({pct:.1f}%) found — deduplication recommended."
    elif issue == 'constant_column':
        col = ev.get('column', 'column')
        val = ev.get('value', 'constant')
        return f"{col} is constant ({val}) — no information content, consider dropping."
    return f"Data quality issue: {issue}."


def _summarize_generic(ev: Dict) -> str:
    return f"Finding detected with evidence: {ev}"


def generate_chapter_summary(chapter_id: str, findings: List[Dict], profile: Dict) -> str:
    """Generate a chapter-level summary from findings."""
    if not findings:
        return f"No significant findings in this chapter."

    summaries = [generate_finding_summary(f) for f in findings]
    count = len(findings)

    chapter_names = {
        'big_picture': 'Big Picture',
        'opportunities': 'Hidden Opportunities',
        'anomalies': 'Unexpected Anomalies',
        'actions': 'Recommended Actions'
    }

    name = chapter_names.get(chapter_id, chapter_id)
    return (f"**{name} Chapter** — {count} key finding{'s' if count != 1 else ''}:\n\n" +
            "\n\n".join(f"• {s}" for s in summaries))


def generate_executive_summary(findings: Dict[str, List], profile: Dict) -> str:
    """Generate an executive summary across all chapters."""
    total_findings = sum(len(v) for v in findings.values())
    if total_findings == 0:
        return "Analysis complete. No significant patterns detected in this dataset."

    lines = [
        f"**DataStory AI Analysis Complete**",
        f"Dataset: {profile.get('shape', [0, 0])[0]:,} rows × {profile.get('shape', [0, 0])[1]} columns",
        f"Quality Score: {profile.get('quality_score', {}).get('overall', 'N/A')}/100",
        f"Total Findings: {total_findings} across 4 chapters",
        ""
    ]

    for chapter_id, chapter_findings in findings.items():
        if chapter_findings:
            lines.append(f"**{chapter_id.replace('_', ' ').title()}**: {len(chapter_findings)} finding(s)")

    lines.append("")
    lines.append("Upload your CSV to discover the story in your data!")

    return "\n".join(lines)