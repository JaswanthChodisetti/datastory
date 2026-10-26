"""Correlation mining module - finds significant correlations between variables."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from itertools import combinations


def detect_correlations(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect significant correlations between numeric columns.

    Returns:
        List of finding dictionaries with evidence
    """
    findings = []

    numeric_cols = list(profile.get('numeric_summary', {}).keys())
    if len(numeric_cols) < 2:
        return findings

    # Get numeric dataframe
    numeric_df = df[numeric_cols].select_dtypes(include='number')

    # Pearson correlation
    pearson_corr = numeric_df.corr(method='pearson')
    # Spearman correlation (for non-linear monotonic relationships)
    spearman_corr = numeric_df.corr(method='spearman')

    # Find significant correlations
    findings.extend(_find_significant_correlations(
        pearson_corr, numeric_df, 'pearson', min_abs_corr=0.4, max_p=0.05
    ))
    findings.extend(_find_significant_correlations(
        spearman_corr, numeric_df, 'spearman', min_abs_corr=0.4, max_p=0.05
    ))

    # Remove duplicates (same pair, different method)
    seen_pairs = set()
    unique_findings = []
    for f in findings:
        pair = tuple(sorted([f['evidence']['column_1'], f['evidence']['column_2']]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_findings.append(f)

    # Sort by absolute correlation strength
    unique_findings.sort(key=lambda x: abs(x['evidence']['correlation']), reverse=True)
    return unique_findings[:5]  # Top 5 correlations


def _find_significant_correlations(corr_matrix: pd.DataFrame, numeric_df: pd.DataFrame,
                                    method: str, min_abs_corr: float = 0.4, max_p: float = 0.05) -> List[Dict]:
    """Find significant correlations from a correlation matrix."""
    findings = []
    cols = corr_matrix.columns

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if j <= i:
                continue

            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) < min_abs_corr:
                continue

            # Calculate p-value
            n = len(numeric_df[[col1, col2]].dropna())
            if n < 10:
                continue

            # t-statistic for correlation - handle perfect correlation edge case
            if abs(corr_value) >= 1.0:
                p_value = 0.0  # Perfect correlation
            else:
                t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

            if p_value > max_p:
                continue

            # Determine relationship type
            if method == 'pearson':
                rel_type = 'linear'
            else:
                rel_type = 'monotonic'

            direction = 'positive' if corr_value > 0 else 'negative'

            evidence = {
                'column_1': col1,
                'column_2': col2,
                'correlation': float(corr_value),
                'method': method,
                'p_value': float(p_value),
                'sample_size': n,
                'relationship_type': rel_type,
                'direction': direction
            }

            # Prepare chart data
            chart_data = numeric_df[[col1, col2]].dropna()
            chart_config = {
                'x': col1,
                'y': col2,
                'title': f'{col1} vs {col2} ({method.capitalize()} r={corr_value:.3f})',
                'trendline': 'ols'
            }

            findings.append({
                'type': 'correlation',
                'title': f'{direction.title()} {rel_type.capitalize()} Correlation: {col1} ↔ {col2}',
                'description': f'{col1} and {col2} have a {direction} {rel_type} relationship '
                             f'({method} r={corr_value:.3f}, p={p_value:.4f}, n={n}). '
                             f'As {col1} increases, {col2} tends to {"increase" if corr_value > 0 else "decrease"}.',
                'evidence': evidence,
                'chart_data': chart_data,
                'chart_type': 'scatter',
                'chart_config': chart_config,
                'score': float(abs(corr_value) * (1 - p_value))
            })

    return findings


def detect_categorical_numeric_relationships(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect relationships between categorical and numeric columns (ANOVA-style).
    """
    findings = []

    numeric_cols = list(profile.get('numeric_summary', {}).keys())
    categorical_cols = [c['Column'] for c in profile['columns'] if c['Semantic Type'] == 'categorical']

    if not numeric_cols or not categorical_cols:
        return findings

    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            # Skip if too many categories
            n_categories = df[cat_col].nunique()
            if n_categories > 20 or n_categories < 2:
                continue

            # Perform ANOVA (Kruskal-Wallis for non-parametric)
            groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
            groups = [g for g in groups if len(g) > 3]

            if len(groups) < 2:
                continue

            try:
                h_stat, p_value = stats.kruskal(*groups)
                if p_value < 0.01 and len(groups) >= 2:
                    # Find which groups differ most
                    group_means = {name: group[num_col].mean() for name, group in df.groupby(cat_col)}
                    max_group = max(group_means, key=group_means.get)
                    min_group = min(group_means, key=group_means.get)
                    diff_pct = (group_means[max_group] - group_means[min_group]) / abs(group_means[min_group]) * 100 if group_means[min_group] != 0 else 0

                    evidence = {
                        'numeric_column': num_col,
                        'categorical_column': cat_col,
                        'kruskal_h': float(h_stat),
                        'p_value': float(p_value),
                        'n_groups': len(groups),
                        'group_means': {k: float(v) for k, v in group_means.items()},
                        'max_group': max_group,
                        'min_group': min_group,
                        'difference_pct': float(diff_pct)
                    }

                    # Box plot chart data
                    chart_data = df[[cat_col, num_col]].dropna()
                    chart_config = {
                        'x': cat_col,
                        'y': num_col,
                        'title': f'{num_col} by {cat_col} (p={p_value:.4f})'
                    }

                    findings.append({
                        'type': 'categorical_numeric',
                        'title': f'{num_col} varies significantly by {cat_col}',
                        'description': f'{num_col} differs significantly across {cat_col} categories '
                                     f'(Kruskal-Wallis H={h_stat:.2f}, p={p_value:.4f}). '
                                     f'{max_group} has the highest mean ({group_means[max_group]:.2f}), '
                                     f'{min_group} the lowest ({group_means[min_group]:.2f}) — a {diff_pct:.0f}% difference.',
                        'evidence': evidence,
                        'chart_data': chart_data,
                        'chart_type': 'box',
                        'chart_config': chart_config,
                        'score': float((1 - p_value) * min(abs(diff_pct) / 100, 1))
                    })
            except Exception:
                continue

    findings.sort(key=lambda x: x['score'], reverse=True)
    return findings[:3]