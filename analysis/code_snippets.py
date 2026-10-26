"""Code snippets for 'Show Calculation' feature - exact code used to compute findings."""

# ============================================================
# Profiling Code Snippets
# ============================================================

PROFILE_CODE = '''
# Data Profiling
import pandas as pd

# Load data
df = pd.read_csv("your_file.csv")

# Basic shape and types
shape = df.shape                    # (rows, columns)
dtypes = df.dtypes.to_dict()        # Column data types
memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

# Duplicates
duplicates = df.duplicated().sum()

# Numeric summary statistics
numeric_cols = df.select_dtypes(include='number').columns
numeric_summary = df[numeric_cols].describe().to_dict()

# Categorical summary
categorical_cols = df.select_dtypes(include='object').columns
categorical_summary = {}
for col in categorical_cols:
    vc = df[col].value_counts()
    categorical_summary[col] = {
        'unique': df[col].nunique(),
        'top_values': vc.head(10).to_dict(),
        'top_pct': (vc.iloc[0] / len(df) * 100) if len(vc) > 0 else 0
    }

# Quality score (completeness, uniqueness, consistency, validity)
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isnull().sum().sum()
completeness = max(0, 100 - (missing_cells / total_cells * 100))
uniqueness = max(0, 100 - (duplicates / len(df) * 100 * 10))
# ... weighted average for overall score
'''

# ============================================================
# Anomaly Detection Code Snippets
# ============================================================

IQR_OUTLIER_CODE = '''
# IQR (Interquartile Range) Outlier Detection
import pandas as pd
import numpy as np

series = df[col].dropna()

Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
outlier_count = len(outliers)
outlier_pct = outlier_count / len(df) * 100

# Results
print(f"Column: {col}")
print(f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Outliers: {outlier_count} ({outlier_pct:.1f}%)")
'''

ZSCORE_OUTLIER_CODE = '''
# Z-Score Outlier Detection
import pandas as pd
import numpy as np
from scipy import stats

series = df[col].dropna()
mean = series.mean()
std = series.std()

z_scores = np.abs((series - mean) / std)
threshold = 3.0
outliers = df.loc[series.index[z_scores > threshold]]
outlier_count = len(outliers)
max_z = z_scores.max()

print(f"Column: {col}")
print(f"Mean={mean:.2f}, Std={std:.2f}")
print(f"Threshold: Z > {threshold}")
print(f"Max Z-score: {max_z:.2f}")
print(f"Outliers: {outlier_count}")
'''

SUDDEN_CHANGE_CODE = '''
# Sudden Change Detection in Time Series
import pandas as pd
import numpy as np

# Sort by datetime
df_sorted = df.sort_values(datetime_col).copy()
df_sorted[datetime_col] = pd.to_datetime(df_sorted[datetime_col])

values = df_sorted[numeric_col].values
diffs = np.diff(values)
pct_changes = np.abs(diffs / values[:-1]) * 100

# Threshold: 3 standard deviations from mean absolute difference
mean_diff = np.mean(np.abs(diffs))
std_diff = np.std(diffs)
threshold = mean_diff + 3 * std_diff

sudden_indices = np.where(np.abs(diffs) > threshold)[0]

for idx in sudden_indices:
    change_pct = pct_changes[idx]
    direction = 'increase' if diffs[idx] > 0 else 'decrease'
    print(f"Sudden {direction} of {change_pct:.1f}% at index {idx}")
'''

QUALITY_ISSUE_CODE = '''
# Data Quality Issue Detection
import pandas as pd

# High missingness
for col in df.columns:
    missing_pct = df[col].isnull().mean() * 100
    if missing_pct > 50:
        print(f"HIGH MISSINGNESS: {col} = {missing_pct:.1f}%")

# Duplicates
dup_count = df.duplicated().sum()
dup_pct = dup_count / len(df) * 100
if dup_count > 0:
    print(f"DUPLICATES: {dup_count} rows ({dup_pct:.1f}%)")

# Constant columns
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"CONSTANT COLUMN: {col} = {df[col].iloc[0]}")
'''

# ============================================================
# Trend Detection Code Snippets
# ============================================================

TREND_CODE = '''
# Trend Detection (Mann-Kendall + Linear Regression)
import pandas as pd
import numpy as np
from scipy import stats

# Sort by datetime
df_sorted = df.sort_values(datetime_col).copy()
df_sorted[datetime_col] = pd.to_datetime(df_sorted[datetime_col])
df_sorted = df_sorted.dropna(subset=[datetime_col, numeric_col])

# Convert time to numeric (Unix timestamp)
time_numeric = df_sorted[datetime_col].astype('int64') // 10**9
values = df_sorted[numeric_col].values

# Mann-Kendall test
def mann_kendall(data):
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if data[j] > data[i]: s += 1
            elif data[j] < data[i]: s -= 1

    unique, counts = np.unique(data, return_counts=True)
    tie_corr = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_corr) / 18

    if var_s == 0: return {'tau': 0, 'p_value': 1.0}

    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return {'tau': tau, 'p_value': p}

mk = mann_kendall(values)
slope, intercept, r, p, se = stats.linregress(time_numeric, values)

direction = 'increasing' if slope > 0 else 'decreasing'
print(f"Mann-Kendall: τ={mk['tau']:.3f}, p={mk['p_value']:.4f}")
print(f"Linear: slope={slope:.4f}, R²={r**2:.3f}, p={p:.4f}")
print(f"Trend: {direction}, strength={abs(r):.0%}")
'''

SEQUENTIAL_TREND_CODE = '''
# Sequential Trend (Row Order) - Mann-Kendall Only
import numpy as np
from scipy import stats

values = df[numeric_col].dropna().values

def mann_kendall(data):
    n = len(data)
    s = sum(1 if data[j] > data[i] else -1 if data[j] < data[i] else 0
            for i in range(n-1) for j in range(i+1, n))
    unique, counts = np.unique(data, return_counts=True)
    tie_corr = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_corr) / 18
    if var_s == 0: return {'tau': 0, 'p_value': 1.0}
    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return {'tau': tau, 'p_value': p}

mk = mann_kendall(values)
direction = 'increasing' if mk['tau'] > 0 else 'decreasing'
print(f"Sequential trend: τ={mk['tau']:.3f}, p={mk['p_value']:.4f}")
print(f"Direction: {direction}")
'''

# ============================================================
# Correlation Code Snippets
# ============================================================

CORRELATION_CODE = '''
# Correlation Detection (Pearson & Spearman)
import pandas as pd
import numpy as np
from scipy import stats

numeric_df = df.select_dtypes(include='number')
pearson = numeric_df.corr(method='pearson')
spearman = numeric_df.corr(method='spearman')

# Find significant correlations
for i, col1 in enumerate(numeric_df.columns):
    for j, col2 in enumerate(numeric_df.columns):
        if j <= i: continue

        for method_name, corr_matrix in [('pearson', pearson), ('spearman', spearman)]:
            r = corr_matrix.loc[col1, col2]
            if abs(r) < 0.4: continue

            n = len(numeric_df[[col1, col2]].dropna())
            if n < 10: continue

            if abs(r) >= 1.0:
                p = 0.0
            else:
                t = r * np.sqrt((n - 2) / (1 - r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), n - 2))

            if p < 0.05:
                direction = 'positive' if r > 0 else 'negative'
                rel_type = 'linear' if method_name == 'pearson' else 'monotonic'
                print(f"{col1} ↔ {col2}: {direction} {rel_type} r={r:.3f}, p={p:.4f}, n={n}")
'''

CAT_NUM_CODE = '''
# Categorical-Numeric Relationship (Kruskal-Wallis)
import pandas as pd
from scipy import stats

for num_col in numeric_cols:
    for cat_col in categorical_cols:
        if df[cat_col].nunique() > 20: continue

        groups = [g[num_col].dropna().values for _, g in df.groupby(cat_col)]
        groups = [g for g in groups if len(g) > 3]
        if len(groups) < 2: continue

        h_stat, p = stats.kruskal(*groups)
        if p < 0.01:
            group_means = {name: g[num_col].mean() for name, g in df.groupby(cat_col)}
            max_g = max(group_means, key=group_means.get)
            min_g = min(group_means, key=group_means.get)
            diff_pct = (group_means[max_g] - group_means[min_g]) / abs(group_means[min_g]) * 100
            print(f"{num_col} by {cat_col}: H={h_stat:.2f}, p={p:.4f}")
            print(f"  {max_g} ({group_means[max_g]:.2f}) vs {min_g} ({group_means[min_g]:.2f}) = {diff_pct:.0f}% diff")
'''

# ============================================================
# Segment Discovery Code Snippets
# ============================================================

SEGMENT_CODE = '''
# Segment Discovery (High/Low Performing Groups)
import pandas as pd
import numpy as np

for num_col in numeric_cols:
    for cat_col in categorical_cols:
        if df[cat_col].nunique() > 15: continue

        grouped = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
        grouped = grouped[grouped['count'] >= 5]
        if len(grouped) < 2: continue

        overall_mean = df[num_col].mean()
        overall_std = df[num_col].std()
        grouped['z_score'] = (grouped['mean'] - overall_mean) / overall_std if overall_std > 0 else 0

        top = grouped.loc[grouped['z_score'].idxmax()]
        bottom = grouped.loc[grouped['z_score'].idxmin()]

        for seg, label in [(top, 'high'), (bottom, 'low')]:
            if abs(seg['z_score']) < 1.5: continue
            pct = (seg['mean'] - overall_mean) / abs(overall_mean) * 100
            direction = 'higher' if seg['z_score'] > 0 else 'lower'
            print(f"{cat_col}={seg[cat_col]}: {direction} {num_col} ({seg['mean']:.2f} vs {overall_mean:.2f}, {pct:+.1f}%, n={int(seg['count'])}, z={seg['z_score']:.2f})")
'''

SEGMENT_2D_CODE = '''
# 2D Segment Discovery (Two Categorical Columns)
import pandas as pd
import numpy as np
from itertools import combinations

for cat1, cat2 in combinations(categorical_cols, 2):
    if df[cat1].nunique() * df[cat2].nunique() > 50: continue

    for num_col in numeric_cols[:2]:
        grouped = df.groupby([cat1, cat2])[num_col].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 5]
        if len(grouped) < 4: continue

        overall_mean = df[num_col].mean()
        overall_std = df[num_col].std()
        grouped['z_score'] = (grouped['mean'] - overall_mean) / overall_std

        top = grouped.loc[grouped['z_score'].idxmax()]
        bottom = grouped.loc[grouped['z_score'].idxmin()]

        for row, label in [(top, 'highest'), (bottom, 'lowest')]:
            if abs(row['z_score']) < 2: continue
            pct = (row['mean'] - overall_mean) / abs(overall_mean) * 100
            print(f"{cat1}={row[cat1]} × {cat2}={row[cat2]}: {label} {num_col}")
            print(f"  Mean: {row['mean']:.2f} vs overall {overall_mean:.2f} ({pct:+.1f}%, n={int(row['count'])})")
'''

# ============================================================
# Mapping from finding type to code snippet
# ============================================================

FINDING_CODE_MAP = {
    # Profiling
    'profile': PROFILE_CODE,

    # Anomalies
    'outlier_iqr': IQR_OUTLIER_CODE,
    'outlier_zscore': ZSCORE_OUTLIER_CODE,
    'sudden_change': SUDDEN_CHANGE_CODE,
    'quality_issue': QUALITY_ISSUE_CODE,

    # Trends
    'trend': TREND_CODE,
    'sequential_trend': SEQUENTIAL_TREND_CODE,

    # Correlations
    'correlation': CORRELATION_CODE,
    'categorical_numeric': CAT_NUM_CODE,

    # Segments
    'segment': SEGMENT_CODE,
    'segment_2d': SEGMENT_2D_CODE,
}


def get_code_snippet(finding_type: str) -> str:
    """Get the code snippet for a finding type."""
    return FINDING_CODE_MAP.get(finding_type, '# No code snippet available for this finding type')


def get_all_code_snippets() -> dict:
    """Get all available code snippets."""
    return FINDING_CODE_MAP.copy()