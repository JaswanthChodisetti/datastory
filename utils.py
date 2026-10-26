
import io

import pandas as pd


def load_csv_from_upload(uploaded_file):
    """Load an uploaded CSV file into a Pandas DataFrame."""
    if uploaded_file is None:
        raise ValueError("No CSV file was uploaded.")

    try:
        df = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin-1")

    if df.empty:
        raise ValueError("The uploaded CSV contains no data.")

    # Stable identifiers for tracing records in the evidence panel.
    df = df.reset_index(drop=True)
    df.insert(0, "_source_row", range(1, len(df) + 1))

    return df


def get_column_info(df):
    """Return useful metadata about each column."""
    info = []

    for column in df.columns:
        series = df[column]

        info.append({
            "column": column,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_percent": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_values": int(series.nunique()),
        })

    return pd.DataFrame(info)


def generate_executive_summary(df):
    """Generate a deterministic summary of the dataset."""
    rows, columns = df.shape

    lines = [
        f"The dataset contains {rows:,} rows and {columns} columns."
    ]

    missing_cells = int(df.isna().sum().sum())
    total_cells = df.size

    if total_cells:
        missing_percent = missing_cells / total_cells * 100
        lines.append(
            f"{missing_cells:,} cells are missing "
            f"({missing_percent:.2f}% of all cells)."
        )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns
    ]

    lines.append(
        f"Numeric columns: {len(numeric_columns)}. "
        f"Other columns: {len(categorical_columns)}."
    )

    if numeric_columns:
        lines.append(
            "Numeric columns include: "
            + ", ".join(map(str, numeric_columns[:8]))
            + (" ..." if len(numeric_columns) > 8 else "")
            + "."
        )

    return "\n".join(lines)
