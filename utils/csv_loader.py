"""Robust CSV loading with auto-detection of encoding, delimiter, and header."""

import pandas as pd
import chardet
from io import StringIO, BytesIO
from typing import Tuple, Optional, List
import streamlit as st


def detect_encoding(file_bytes: bytes) -> str:
    """Detect file encoding using chardet."""
    result = chardet.detect(file_bytes)
    encoding = result.get('encoding', 'utf-8')
    confidence = result.get('confidence', 0)

    # Fallback for low confidence
    if confidence < 0.7:
        encoding = 'utf-8'

    return encoding


def detect_delimiter(sample: str) -> str:
    """Detect delimiter from a sample of the file."""
    delimiters = [',', ';', '\t', '|', ':']
    delimiter_counts = {d: sample.count(d) for d in delimiters}
    return max(delimiter_counts, key=delimiter_counts.get)


def load_csv(file_bytes: bytes, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """
    Load CSV with auto-detection of encoding, delimiter, and header.

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    metadata = {
        'encoding': None,
        'delimiter': None,
        'header_row': None,
        'rows': 0,
        'cols': 0,
        'warnings': []
    }

    # Detect encoding
    encoding = detect_encoding(file_bytes)
    metadata['encoding'] = encoding

    # Decode bytes to string
    try:
        text = file_bytes.decode(encoding)
    except UnicodeDecodeError:
        # Fallback encodings
        for fallback in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = file_bytes.decode(fallback)
                encoding = fallback
                metadata['encoding'] = encoding
                metadata['warnings'].append(f"Used fallback encoding: {fallback}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode file with any common encoding")

    # Detect delimiter from first few lines
    lines = text.split('\n')[:10]
    sample = '\n'.join(lines)
    delimiter = detect_delimiter(sample)
    metadata['delimiter'] = delimiter

    # Try to find header row (first row that looks like headers)
    header_row = 0
    for i, line in enumerate(lines[:5]):
        if delimiter in line and not line.strip().startswith('#'):
            # Check if this row has consistent column count with subsequent rows
            header_row = i
            break
    metadata['header_row'] = header_row

    # Load with pandas
    try:
        df = pd.read_csv(
            StringIO(text),
            delimiter=delimiter,
            header=header_row,
            encoding=encoding,
            on_bad_lines='skip',
            **kwargs
        )
    except Exception as e:
        # Fallback: try without header
        try:
            df = pd.read_csv(
                StringIO(text),
                delimiter=delimiter,
                header=None,
                encoding=encoding,
                on_bad_lines='skip',
                **kwargs
            )
            metadata['warnings'].append("Could not detect header, using default integer index")
            metadata['header_row'] = None
        except Exception as e2:
            raise ValueError(f"Failed to parse CSV: {e2}")

    metadata['rows'] = len(df)
    metadata['cols'] = len(df.columns)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Handle duplicate column names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_indices[1:], 1):
            cols[idx] = f"{dup}_{i}"
    df.columns = cols

    return df, metadata


def load_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """Load CSV from Streamlit uploaded file object."""
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for potential re-reads
    return load_csv(file_bytes)


def get_sample_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Get a sample of the dataframe for preview."""
    return df.head(n)


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Get detailed column information."""
    info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        unique = df[col].nunique()

        # Try to infer semantic type
        semantic_type = infer_semantic_type(df[col])

        info.append({
            'Column': col,
            'Type': dtype,
            'Non-Null': non_null,
            'Null %': f"{(null_count / len(df) * 100):.1f}%",
            'Unique': unique,
            'Semantic Type': semantic_type
        })

    return pd.DataFrame(info)


def infer_semantic_type(series: pd.Series) -> str:
    """Infer semantic type of a column."""
    # Drop nulls for analysis
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'empty'

    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if it looks like an ID (many unique values, sequential)
        if non_null.nunique() == len(non_null) and non_null.nunique() > 10:
            # Check if roughly sequential
            try:
                diff = non_null.sort_values().diff().dropna()
                if diff.nunique() <= 3 and (diff == 1).mean() > 0.8:
                    return 'id'
            except:
                pass
        return 'numeric'

    # Check if datetime
    try:
        pd.to_datetime(non_null.iloc[0])
        # Try parsing a sample
        parsed = pd.to_datetime(non_null.head(100), errors='coerce')
        if parsed.notna().mean() > 0.8:
            return 'datetime'
    except:
        pass

    # Check if categorical (low cardinality)
    unique_ratio = non_null.nunique() / len(non_null)
    if unique_ratio < 0.05 and non_null.nunique() < 50:
        return 'categorical'

    # Check if text (high cardinality strings)
    if unique_ratio > 0.5:
        return 'text'

    return 'categorical'