"""Trend detection module - identifies trends in time series and ordered data."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
from utils.csv_loader import infer_semantic_type


def detect_trends(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect trends in the dataframe.

    Looks for:
    - Time series trends (Mann-Kendall test, linear regression slope)
    - Sequential trends in ordered numeric data
    - Monotonic trends

    Returns:
        List of finding dictionaries with evidence
    """
    findings = []

    datetime_cols = profile.get('datetime_columns', [])
    numeric_cols = list(profile.get('numeric_summary', {}).keys())

    if not datetime_cols or not numeric_cols:
        return findings

    # For each datetime column, check trends in numeric columns
    for dt_col in datetime_cols:
        try:
            # Parse datetime
            df_sorted = df.sort_values(dt_col).copy()
            df_sorted[dt_col] = pd.to_datetime(df_sorted[dt_col], errors='coerce')
            df_sorted = df_sorted.dropna(subset=[dt_col])

            if len(df_sorted) < 10:
                continue

            # Convert time to numeric for correlation
            time_numeric = df_sorted[dt_col].astype('int64') // 10**9

            for num_col in numeric_cols:
                if num_col not in df_sorted.columns:
                    continue

                values = df_sorted[num_col].dropna()
                if len(values) < 10:
                    continue

                # Align time and values
                valid_idx = df_sorted[num_col].notna()
                time_valid = time_numeric[valid_idx]
                vals_valid = df_sorted.loc[valid_idx, num_col]

                if len(vals_valid) < 10:
                    continue

                # Mann-Kendall trend test
                mk_result = mann_kendall_test(vals_valid.values)

                # Linear regression slope
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_valid, vals_valid)

                # Determine trend direction and strength
                trend_direction = 'increasing' if slope > 0 else 'decreasing'
                trend_strength = abs(r_value)

                # Only report significant trends
                if mk_result['p_value'] < 0.05 and trend_strength > 0.3:
                    # Get evidence rows
                    evidence = {
                        'datetime_column': dt_col,
                        'numeric_column': num_col,
                        'mann_kendall_tau': float(mk_result['tau']),
                        'mann_kendall_p': float(mk_result['p_value']),
                        'linear_slope': float(slope),
                        'linear_r_squared': float(r_value ** 2),
                        'linear_p_value': float(p_value),
                        'trend_direction': trend_direction,
                        'trend_strength': float(trend_strength),
                        'start_value': float(vals_valid.iloc[0]),
                        'end_value': float(vals_valid.iloc[-1]),
                        'change_pct': float((vals_valid.iloc[-1] - vals_valid.iloc[0]) / abs(vals_valid.iloc[0]) * 100) if vals_valid.iloc[0] != 0 else 0,
                        'n_points': len(vals_valid),
                        'start_date': str(df_sorted[dt_col].iloc[0]),
                        'end_date': str(df_sorted[dt_col].iloc[-1])
                    }

                    # Prepare chart data
                    chart_data = df_sorted[[dt_col, num_col]].dropna()
                    chart_config = {
                        'x': dt_col,
                        'y': num_col,
                        'title': f'{num_col} over {dt_col}',
                        'trendline': 'ols'
                    }

                    findings.append({
                        'type': 'trend',
                        'title': f'{trend_direction.title()} Trend in {num_col}',
                        'description': f'{num_col} shows a {trend_direction} trend over {dt_col} '
                                     f'({trend_strength:.0%} correlation, p={mk_result["p_value"]:.3f}). '
                                     f'Value changed by {evidence["change_pct"]:.1f}% from {evidence["start_date"]} to {evidence["end_date"]}.',
                        'evidence': evidence,
                        'chart_data': chart_data,
                        'chart_type': 'scatter',
                        'chart_config': chart_config,
                        'score': float(trend_strength * (1 - mk_result['p_value']))
                    })
        except Exception as e:
            continue

    # Also check for sequential trends in data without datetime (row order)
    for num_col in numeric_cols:
        values = df[num_col].dropna()
        if len(values) < 20:
            continue

        mk_result = mann_kendall_test(values.values)
        if mk_result['p_value'] < 0.01 and abs(mk_result['tau']) > 0.4:
            evidence = {
                'numeric_column': num_col,
                'mann_kendall_tau': float(mk_result['tau']),
                'mann_kendall_p': float(mk_result['p_value']),
                'trend_direction': 'increasing' if mk_result['tau'] > 0 else 'decreasing',
                'n_points': len(values)
            }

            chart_data = pd.DataFrame({'index': range(len(values)), num_col: values.values})
            chart_config = {
                'x': 'index',
                'y': num_col,
                'title': f'{num_col} Sequential Trend',
                'trendline': 'ols'
            }

            findings.append({
                'type': 'sequential_trend',
                'title': f'Sequential Trend in {num_col}',
                'description': f'{num_col} shows a {evidence["trend_direction"]} trend in row order '
                             f'(Kendall τ={mk_result["tau"]:.3f}, p={mk_result["p_value"]:.3f}).',
                'evidence': evidence,
                'chart_data': chart_data,
                'chart_type': 'scatter',
                'chart_config': chart_config,
                'score': float(abs(mk_result['tau']) * (1 - mk_result['p_value']))
            })

    # Sort by score descending
    findings.sort(key=lambda x: x['score'], reverse=True)
    return findings[:3]  # Top 3 trends


def mann_kendall_test(data: np.ndarray) -> Dict[str, float]:
    """
    Perform Mann-Kendall trend test.
    Returns tau and p-value.
    """
    n = len(data)
    if n < 4:
        return {'tau': 0, 'p_value': 1.0}

    # Calculate S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if data[j] > data[i]:
                s += 1
            elif data[j] < data[i]:
                s -= 1

    # Calculate variance (accounting for ties)
    unique_vals, counts = np.unique(data, return_counts=True)
    tie_correction = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18

    if var_s == 0:
        return {'tau': 0, 'p_value': 1.0}

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    return {'tau': tau, 'p_value': p_value, 'z': z}