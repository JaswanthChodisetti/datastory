"""Anomaly detection module - finds outliers, sudden changes, and data quality issues."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
from utils.csv_loader import infer_semantic_type


def detect_anomalies(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect anomalies in the dataframe.

    Methods:
    - IQR (Interquartile Range) for univariate outliers
    - Z-score for extreme values
    - Sudden changes in time series
    - Data quality issues (unexpected nulls, duplicates, etc.)

    Returns:
        List of finding dictionaries with evidence
    """
    findings = []

    numeric_cols = list(profile.get('numeric_summary', {}).keys())
    datetime_cols = profile.get('datetime_columns', [])

    # 1. IQR-based outliers for numeric columns
    for col in numeric_cols:
        iqr_findings = _detect_iqr_outliers(df, col, profile)
        findings.extend(iqr_findings)

    # 2. Z-score outliers (for comparison)
    for col in numeric_cols:
        zscore_findings = _detect_zscore_outliers(df, col, profile)
        findings.extend(zscore_findings)

    # 3. Sudden changes in time series
    if datetime_cols and numeric_cols:
        for dt_col in datetime_cols:
            for num_col in numeric_cols:
                sudden_findings = _detect_sudden_changes(df, dt_col, num_col, profile)
                findings.extend(sudden_findings)

    # 4. Data quality anomalies
    quality_findings = _detect_quality_issues(df, profile)
    findings.extend(quality_findings)

    # Deduplicate and sort by score
    seen = set()
    unique_findings = []
    for f in findings:
        key = (f['type'], f['evidence'].get('column'), f['evidence'].get('method'))
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    unique_findings.sort(key=lambda x: x['score'], reverse=True)
    return unique_findings[:5]


def _detect_iqr_outliers(df: pd.DataFrame, col: str, profile: Dict) -> List[Dict[str, Any]]:
    """Detect outliers using IQR method."""
    findings = []
    series = df[col].dropna()

    if len(series) < 10:
        return findings

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
        return findings

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = outlier_count / len(df) * 100

    if outlier_count == 0:
        return findings

    # Only report if meaningful number of outliers
    if outlier_pct < 0.5 and outlier_count < 3:
        return findings

    # Get example outliers
    example_low = outliers[outliers[col] < lower_bound][col].min() if (outliers[col] < lower_bound).any() else None
    example_high = outliers[outliers[col] > upper_bound][col].max() if (outliers[col] > upper_bound).any() else None

    evidence = {
        'column': col,
        'method': 'IQR',
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outlier_count': int(outlier_count),
        'outlier_percentage': float(outlier_pct),
        'example_low': float(example_low) if example_low is not None else None,
        'example_high': float(example_high) if example_high is not None else None,
        'row_indices': outliers.index.tolist()[:20]  # First 20 row indices
    }

    # Chart: box plot with outliers highlighted
    chart_data = df[[col]].dropna().copy()
    chart_data['is_outlier'] = chart_data.index.isin(outliers.index)
    chart_config = {
        'y': col,
        'title': f'IQR Outliers in {col} ({outlier_count} outliers, {outlier_pct:.1f}%)'
    }

    # Format extreme values description
    extreme_parts = []
    if example_low is not None:
        extreme_parts.append(f"low={example_low:.2f}")
    if example_high is not None:
        extreme_parts.append(f"high={example_high:.2f}")
    extreme_desc = ", ".join(extreme_parts) if extreme_parts else "none detected"

    findings.append({
            'type': 'outlier_iqr',
            'title': f'IQR Outliers in {col}',
            'description': f'Found {outlier_count} outliers ({outlier_pct:.1f}%) in {col} using IQR method. '
                         f'Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]. '
                         f'Extreme values: {extreme_desc}.',
            'evidence': evidence,
            'chart_data': chart_data,
            'chart_type': 'box',
            'chart_config': chart_config,
            'score': float(min(outlier_pct / 5, 1) * 0.8)
        })

    return findings


def _detect_zscore_outliers(df: pd.DataFrame, col: str, profile: Dict, threshold: float = 3.0) -> List[Dict[str, Any]]:
    """Detect outliers using Z-score method."""
    findings = []
    series = df[col].dropna()

    if len(series) < 10:
        return findings

    mean = series.mean()
    std = series.std()

    if std == 0:
        return findings

    z_scores = np.abs((series - mean) / std)
    outliers = df.loc[series.index[z_scores > threshold]]
    outlier_count = len(outliers)
    outlier_pct = outlier_count / len(df) * 100

    if outlier_count == 0:
        return findings

    # Only report if different from IQR or more extreme
    max_z = z_scores.max()

    evidence = {
        'column': col,
        'method': 'Z-score',
        'mean': float(mean),
        'std': float(std),
        'threshold': threshold,
        'max_z_score': float(max_z),
        'outlier_count': int(outlier_count),
        'outlier_percentage': float(outlier_pct),
        'row_indices': outliers.index.tolist()[:20]
    }

    # Only add if it adds value beyond IQR
    if max_z > 4 or outlier_count > 5:
        findings.append({
            'type': 'outlier_zscore',
            'title': f'Extreme Z-Score Outliers in {col}',
            'description': f'Found {outlier_count} extreme outliers ({outlier_pct:.1f}%) in {col} '
                         f'with Z-score > {threshold} (max Z = {max_z:.2f}).',
            'evidence': evidence,
            'chart_data': df[[col]].dropna(),
            'chart_type': 'box',
            'chart_config': {'y': col, 'title': f'Z-Score Outliers in {col}'},
            'score': float(min(max_z / 10, 1) * 0.6)
        })

    return findings


def _detect_sudden_changes(df: pd.DataFrame, dt_col: str, num_col: str, profile: Dict) -> List[Dict[str, Any]]:
    """Detect sudden changes/jumps in time series."""
    findings = []

    try:
        df_sorted = df.sort_values(dt_col).copy()
        df_sorted[dt_col] = pd.to_datetime(df_sorted[dt_col], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[dt_col, num_col])

        if len(df_sorted) < 10:
            return findings

        values = df_sorted[num_col].values
        dates = df_sorted[dt_col].values

        # Calculate differences
        diffs = np.diff(values)
        pct_changes = np.abs(diffs / values[:-1]) * 100

        # Find large changes (> 3 std deviations from mean diff)
        mean_diff = np.mean(np.abs(diffs))
        std_diff = np.std(diffs)

        if std_diff == 0:
            return findings

        # Threshold: 3 standard deviations
        threshold = mean_diff + 3 * std_diff
        sudden_indices = np.where(np.abs(diffs) > threshold)[0]

        if len(sudden_indices) == 0:
            return findings

        for idx in sudden_indices[:3]:  # Top 3 sudden changes
            change_pct = pct_changes[idx] if values[idx] != 0 else 0
            direction = 'increase' if diffs[idx] > 0 else 'decrease'

            evidence = {
                'datetime_column': dt_col,
                'numeric_column': num_col,
                'change_index': int(idx),
                'date_before': str(dates[idx]),
                'date_after': str(dates[idx + 1]),
                'value_before': float(values[idx]),
                'value_after': float(values[idx + 1]),
                'absolute_change': float(diffs[idx]),
                'percent_change': float(change_pct),
                'direction': direction,
                'threshold_used': float(threshold)
            }

            chart_data = df_sorted[[dt_col, num_col]].copy()
            chart_config = {
                'x': dt_col,
                'y': num_col,
                'title': f'Sudden Change in {num_col} on {dates[idx]}'
            }

            findings.append({
                'type': 'sudden_change',
                'title': f'Sudden {direction.title()} in {num_col}',
                'description': f'{num_col} had a sudden {direction} of {change_pct:.1f}% '
                             f'between {dates[idx]} ({values[idx]:.2f}) and {dates[idx+1]} ({values[idx+1]:.2f}).',
                'evidence': evidence,
                'chart_data': chart_data,
                'chart_type': 'line',
                'chart_config': chart_config,
                'score': float(min(change_pct / 100, 1) * 0.9)
            })

    except Exception:
        pass

    return findings


def _detect_quality_issues(df: pd.DataFrame, profile: Dict) -> List[Dict[str, Any]]:
    """Detect data quality issues."""
    findings = []

    # High missingness in columns
    for col, missing_pct in profile['missing_pct'].items():
        if missing_pct > 50:
            evidence = {
                'column': col,
                'issue_type': 'high_missingness',
                'missing_percentage': float(missing_pct),
                'missing_count': int(profile['missing'][col]),
                'total_rows': profile['shape'][0]
            }
            findings.append({
                'type': 'quality_issue',
                'title': f'High Missingness in {col}',
                'description': f'{col} has {missing_pct:.1f}% missing values ({profile["missing"][col]:,} of {profile["shape"][0]:,} rows).',
                'evidence': evidence,
                'chart_data': None,
                'chart_type': None,
                'chart_config': None,
                'score': float(missing_pct / 100 * 0.7)
            })

    # Duplicate rows
    dup_count = profile['duplicates']
    if dup_count > 0:
        dup_pct = dup_count / profile['shape'][0] * 100
        evidence = {
            'issue_type': 'duplicates',
            'duplicate_count': int(dup_count),
            'duplicate_percentage': float(dup_pct)
        }
        findings.append({
            'type': 'quality_issue',
            'title': 'Duplicate Rows Detected',
            'description': f'Found {dup_count:,} duplicate rows ({dup_pct:.1f}% of data). Consider deduplication.',
            'evidence': evidence,
            'chart_data': None,
            'chart_type': None,
            'chart_config': None,
            'score': float(min(dup_pct / 10, 1) * 0.5)
        })

    # Constant columns (zero variance)
    for col in df.columns:
        if df[col].nunique() == 1:
            evidence = {
                'column': col,
                'issue_type': 'constant_column',
                'value': str(df[col].iloc[0])
            }
            findings.append({
                'type': 'quality_issue',
                'title': f'Constant Column: {col}',
                'description': f'{col} has the same value ({df[col].iloc[0]}) for all rows. No information content.',
                'evidence': evidence,
                'chart_data': None,
                'chart_type': None,
                'chart_config': None,
                'score': 0.3
            })

    return findings