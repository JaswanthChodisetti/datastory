"""Data profiling module - generates comprehensive dataset summary."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from utils.csv_loader import get_column_info, infer_semantic_type


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive profile of the dataframe.

    Returns:
        Dict with profile information including:
        - shape, dtypes, missingness, basic stats, column details
    """
    profile = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'columns': [],
        'numeric_summary': {},
        'categorical_summary': {},
        'datetime_columns': [],
        'id_columns': [],
        'text_columns': []
    }

    # Column-level details
    col_info = get_column_info(df)
    profile['columns'] = col_info.to_dict('records')

    # Separate columns by semantic type
    for col in df.columns:
        sem_type = infer_semantic_type(df[col])
        if sem_type == 'numeric':
            profile['numeric_summary'][col] = describe_numeric(df[col])
            if col not in profile['id_columns']:
                profile['id_columns'].append(col)
        elif sem_type == 'categorical':
            profile['categorical_summary'][col] = describe_categorical(df[col])
        elif sem_type == 'datetime':
            profile['datetime_columns'].append(col)
        elif sem_type == 'text':
            profile['text_columns'].append(col)
        elif sem_type == 'id':
            profile['id_columns'].append(col)

    # Overall data quality score
    profile['quality_score'] = calculate_quality_score(df, profile)

    return profile


def describe_numeric(series: pd.Series) -> Dict[str, Any]:
    """Describe a numeric column with relevant statistics."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {'count': 0, 'error': 'All null'}

    desc = {
        'count': int(non_null.count()),
        'mean': float(non_null.mean()),
        'std': float(non_null.std()),
        'min': float(non_null.min()),
        'max': float(non_null.max()),
        'median': float(non_null.median()),
        'q25': float(non_null.quantile(0.25)),
        'q75': float(non_null.quantile(0.75)),
        'skew': float(non_null.skew()),
        'kurtosis': float(non_null.kurtosis()),
        'zeros': int((non_null == 0).sum()),
        'negative': int((non_null < 0).sum())
    }

    # Add IQR
    desc['iqr'] = desc['q75'] - desc['q25']

    return desc


def describe_categorical(series: pd.Series) -> Dict[str, Any]:
    """Describe a categorical column."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {'count': 0, 'unique': 0, 'error': 'All null'}

    value_counts = non_null.value_counts()
    top_values = value_counts.head(10).to_dict()

    return {
        'count': int(non_null.count()),
        'unique': int(non_null.nunique()),
        'top_values': {str(k): int(v) for k, v in top_values.items()},
        'top_value_pct': float(value_counts.iloc[0] / len(non_null) * 100) if len(value_counts) > 0 else 0,
        'entropy': calculate_entropy(non_null)
    }


def calculate_entropy(series: pd.Series) -> float:
    """Calculate Shannon entropy of a categorical series."""
    value_counts = series.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
    return float(entropy)


def calculate_quality_score(df: pd.DataFrame, profile: Dict) -> Dict[str, Any]:
    """Calculate overall data quality score (0-100)."""
    scores = {}

    # Completeness (no missing values)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    scores['completeness'] = max(0, 100 - (missing_cells / total_cells * 100))

    # Uniqueness (no duplicates)
    dup_pct = profile['duplicates'] / len(df) * 100 if len(df) > 0 else 0
    scores['uniqueness'] = max(0, 100 - dup_pct * 10)

    # Consistency (reasonable dtypes, no mixed types in columns)
    # Simple heuristic: penalize object columns that could be numeric/datetime
    consistency_penalty = 0
    for col in df.select_dtypes(include=['object']).columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            # Check if mostly numeric
            try:
                pd.to_numeric(non_null.head(100))
                consistency_penalty += 5
            except:
                pass
    scores['consistency'] = max(0, 100 - consistency_penalty)

    # Validity (values in expected ranges - placeholder)
    scores['validity'] = 85  # Would need domain rules

    # Overall weighted score
    weights = {'completeness': 0.4, 'uniqueness': 0.2, 'consistency': 0.2, 'validity': 0.2}
    overall = sum(scores[k] * weights[k] for k in weights)

    return {
        'overall': round(overall, 1),
        'components': {k: round(v, 1) for k, v in scores.items()},
        'grade': grade_from_score(overall)
    }


def grade_from_score(score: float) -> str:
    """Convert quality score to letter grade."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'


def get_profile_summary_text(profile: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the profile."""
    lines = []
    lines.append(f"**Dataset Shape:** {profile['shape'][0]:,} rows × {profile['shape'][1]} columns")
    lines.append(f"**Memory Usage:** {profile['memory_usage_mb']:.2f} MB")
    lines.append(f"**Duplicate Rows:** {profile['duplicates']:,}")
    lines.append(f"**Data Quality Score:** {profile['quality_score']['overall']}/100 ({profile['quality_score']['grade']})")

    missing_total = sum(profile['missing'].values())
    if missing_total > 0:
        lines.append(f"**Missing Values:** {missing_total:,} total ({missing_total/profile['shape'][0]/profile['shape'][1]*100:.1f}% of cells)")

    lines.append(f"**Numeric Columns:** {len(profile['numeric_summary'])}")
    lines.append(f"**Categorical Columns:** {len(profile['categorical_summary'])}")
    lines.append(f"**Datetime Columns:** {len(profile['datetime_columns'])}")
    lines.append(f"**Text Columns:** {len(profile['text_columns'])}")
    lines.append(f"**ID Columns:** {len(profile['id_columns'])}")

    return '\n'.join(lines)