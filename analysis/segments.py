"""Segment discovery module - finds high-variance segments and interesting subgroups."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List
from itertools import combinations


def detect_segments(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Discover interesting segments in the data.

    Methods:
    - High variance segments (groupby on categorical, find groups with extreme values)
    - Segment profiling (describe each segment)
    - Cross-segment comparisons

    Returns:
        List of finding dictionaries with evidence
    """
    findings = []

    numeric_cols = list(profile.get('numeric_summary', {}).keys())
    categorical_cols = [c['Column'] for c in profile['columns'] if c['Semantic Type'] == 'categorical']

    if not numeric_cols or not categorical_cols:
        return findings

    # For each numeric column, find segments with extreme values
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            segment_findings = _analyze_segments(df, num_col, cat_col, profile)
            findings.extend(segment_findings)

    # Also look for 2D segments (two categorical columns)
    if len(categorical_cols) >= 2:
        for cat1, cat2 in combinations(categorical_cols, 2):
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric cols
                segment_findings = _analyze_2d_segments(df, num_col, cat1, cat2, profile)
                findings.extend(segment_findings)

    # Sort by score and return top findings
    findings.sort(key=lambda x: x['score'], reverse=True)
    return findings[:4]


def _analyze_segments(df: pd.DataFrame, num_col: str, cat_col: str, profile: Dict) -> List[Dict[str, Any]]:
    """Analyze segments of a numeric column by a categorical column."""
    findings = []

    # Skip if too many categories
    n_categories = df[cat_col].nunique()
    if n_categories > 15 or n_categories < 2:
        return findings

    # Group statistics
    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    grouped = grouped[grouped['count'] >= 5]  # Minimum sample size

    if len(grouped) < 2:
        return findings

    overall_mean = df[num_col].mean()
    overall_std = df[num_col].std()

    # Find segments with significantly different means (z-score > 2)
    grouped['z_score'] = (grouped['mean'] - overall_mean) / overall_std if overall_std > 0 else 0

    # Top and bottom segments
    top_segment = grouped.loc[grouped['z_score'].idxmax()]
    bottom_segment = grouped.loc[grouped['z_score'].idxmin()]

    top_z = top_segment['z_score']
    bottom_z = bottom_segment['z_score']

    # Only report if significant
    if abs(top_z) < 1.5 and abs(bottom_z) < 1.5:
        return findings

    for _, segment in [(top_segment, 'high'), (bottom_segment, 'low')]:
        if abs(segment['z_score']) < 1.5:
            continue

        cat_value = segment[cat_col]
        seg_mean = segment['mean']
        seg_count = int(segment['count'])
        pct_diff = (seg_mean - overall_mean) / abs(overall_mean) * 100 if overall_mean != 0 else 0

        # Get example rows
        segment_rows = df[df[cat_col] == cat_value]
        example_indices = segment_rows.index.tolist()[:10]

        evidence = {
            'numeric_column': num_col,
            'categorical_column': cat_col,
            'segment_value': str(cat_value),
            'segment_type': segment,
            'segment_mean': float(seg_mean),
            'segment_count': seg_count,
            'overall_mean': float(overall_mean),
            'z_score': float(segment['z_score']),
            'pct_difference': float(pct_diff),
            'segment_std': float(segment['std']),
            'example_row_indices': example_indices
        }

        # Chart: bar chart of segment means
        chart_data = grouped[[cat_col, 'mean']].copy()
        chart_data.columns = [cat_col, f'Mean {num_col}']
        chart_config = {
            'x': cat_col,
            'y': f'Mean {num_col}',
            'title': f'Mean {num_col} by {cat_col}',
            'orientation': 'v'
        }

        direction = 'higher' if segment['z_score'] > 0 else 'lower'
        findings.append({
            'type': 'segment',
            'title': f'{cat_value}: {direction.title()} {num_col} Segment',
            'description': f'The segment **{cat_col}={cat_value}** has {direction} {num_col} '
                         f'({seg_mean:.2f} vs overall {overall_mean:.2f}, {pct_diff:+.1f}%, '
                         f'n={seg_count}, z={segment["z_score"]:.2f}).',
            'evidence': evidence,
            'chart_data': chart_data,
            'chart_type': 'bar',
            'chart_config': chart_config,
            'score': float(min(abs(segment['z_score']) / 4, 1) * 0.85)
        })

    return findings


def _analyze_2d_segments(df: pd.DataFrame, num_col: str, cat1: str, cat2: str, profile: Dict) -> List[Dict[str, Any]]:
    """Analyze segments by two categorical columns."""
    findings = []

    # Skip if too many combinations
    n_combos = df[cat1].nunique() * df[cat2].nunique()
    if n_combos > 50:
        return findings

    grouped = df.groupby([cat1, cat2])[num_col].agg(['mean', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= 5]

    if len(grouped) < 4:
        return findings

    overall_mean = df[num_col].mean()
    overall_std = df[num_col].std()

    if overall_std == 0:
        return findings

    grouped['z_score'] = (grouped['mean'] - overall_mean) / overall_std

    # Find most extreme combination
    top_idx = grouped['z_score'].idxmax()
    bottom_idx = grouped['z_score'].idxmin()

    top_row = grouped.loc[top_idx]
    bottom_row = grouped.loc[bottom_idx]

    if abs(top_row['z_score']) < 2 and abs(bottom_row['z_score']) < 2:
        return findings

    for _, row in [(top_row, 'high'), (bottom_row, 'low')]:
        if abs(row['z_score']) < 2:
            continue

        evidence = {
            'numeric_column': num_col,
            'categorical_columns': [cat1, cat2],
            'segment_values': {cat1: str(row[cat1]), cat2: str(row[cat2])},
            'segment_mean': float(row['mean']),
            'segment_count': int(row['count']),
            'overall_mean': float(overall_mean),
            'z_score': float(row['z_score']),
            'pct_difference': float((row['mean'] - overall_mean) / abs(overall_mean) * 100) if overall_mean != 0 else 0
        }

        # Heatmap data
        pivot = grouped.pivot(index=cat1, columns=cat2, values='mean')
        chart_data = pivot
        chart_config = {
            'title': f'Mean {num_col} by {cat1} × {cat2}',
            'colorscale': 'RdBu'
        }

        direction = 'highest' if row['z_score'] > 0 else 'lowest'
        findings.append({
            'type': 'segment_2d',
            'title': f'{row[cat1]} × {row[cat2]}: {direction.title()} {num_col}',
            'description': f'The combination **{cat1}={row[cat1]}** and **{cat2}={row[cat2]}** has the '
                         f'{direction} {num_col} ({row["mean"]:.2f} vs overall {overall_mean:.2f}, '
                         f'{evidence["pct_difference"]:+.1f}%, n={int(row["count"])}).',
            'evidence': evidence,
            'chart_data': chart_data,
            'chart_type': 'heatmap',
            'chart_config': chart_config,
            'score': float(min(abs(row['z_score']) / 5, 1) * 0.8)
        })

    return findings