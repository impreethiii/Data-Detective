"""
profiler.py
Core profiling engine: missing values, duplicates, data types, basic stats.
"""

import pandas as pd
import numpy as np


def _get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Returns column names that hold text/categorical data, without relying on
    select_dtypes(include=["object", "str"]) — some pandas versions raise a
    TypeError when 'object' and 'str' are passed together. Checking dtype
    names directly works across pandas versions.
    """
    return [col for col in df.columns if str(df[col].dtype) in ("object", "str", "string", "category")]


def profile_missing_values(df: pd.DataFrame) -> dict:
    """Returns missing value count and percentage per column."""
    total_rows = len(df)
    missing = df.isnull().sum()
    result = {}
    for col in df.columns:
        count = int(missing[col])
        pct = round((count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        result[col] = {"missing_count": count, "missing_pct": pct}
    return result


def profile_duplicates(df: pd.DataFrame) -> dict:
    """Returns exact duplicate row info."""
    dup_mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated().sum())  # count excluding first occurrence
    total_rows = len(df)
    pct = round((dup_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
    return {
        "duplicate_rows": dup_count,
        "duplicate_pct": pct,
        "duplicate_row_indices": df[dup_mask].index.tolist(),
    }


def profile_data_types(df: pd.DataFrame) -> dict:
    """
    Detects declared dtype vs. inferred 'actual' type by attempting conversions.
    Flags columns where the stored type doesn't match what the values look like
    (e.g. a numeric column stored as text, or dates stored as strings).
    """
    result = {}
    for col in df.columns:
        declared = str(df[col].dtype)
        series = df[col].dropna()

        inferred = declared
        mismatch = False

        # covers both legacy pandas ('object') and pandas 3.x ('str') text dtypes
        if declared in ("object", "str") and len(series) > 0:
            # Try numeric
            numeric_converted = pd.to_numeric(series, errors="coerce")
            numeric_success_rate = numeric_converted.notna().mean()

            # Try datetime
            try:
                datetime_converted = pd.to_datetime(series, errors="coerce", format=None)
                datetime_success_rate = datetime_converted.notna().mean()
            except Exception:
                datetime_success_rate = 0.0

            if numeric_success_rate > 0.9:
                inferred = "numeric (stored as text)"
                mismatch = True
            elif datetime_success_rate > 0.9:
                inferred = "datetime (stored as text)"
                mismatch = True

        result[col] = {
            "declared_dtype": declared,
            "inferred_type": inferred,
            "type_mismatch": mismatch,
        }
    return result


def profile_basic_stats(df: pd.DataFrame) -> dict:
    """Descriptive stats for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue
        result[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }
    return result


def profile_inconsistent_categories(df: pd.DataFrame, similarity_threshold: float = 0.85) -> dict:
    """
    Detects likely-duplicate category values within categorical columns
    (e.g. 'USA' vs 'U.S.A' vs 'usa') using normalized string comparison.
    """
    from difflib import SequenceMatcher

    result = {}
    categorical_cols = _get_categorical_columns(df)

    for col in categorical_cols:
        values = df[col].dropna().astype(str).unique().tolist()
        if len(values) < 2 or len(values) > 200:
            # skip very high-cardinality columns (likely free text / IDs, not categories)
            continue

        # skip columns that are actually dates/numbers stored as text — fuzzy string
        # matching on values like "2024-04-12" vs "2024-04-13" produces false positives
        # since they're similar strings but not the same category at all
        sample = pd.Series(values)
        numeric_like = pd.to_numeric(sample, errors="coerce").notna().mean()
        datetime_like = pd.to_datetime(sample, errors="coerce", format=None).notna().mean()
        if numeric_like > 0.8 or datetime_like > 0.8:
            continue

        clusters = []
        seen = set()
        for i, v1 in enumerate(values):
            if v1 in seen:
                continue
            group = [v1]
            seen.add(v1)
            for v2 in values[i + 1:]:
                if v2 in seen:
                    continue
                norm1, norm2 = v1.lower().strip(), v2.lower().strip()
                # same after normalizing case/whitespace -> definite match (e.g. "South" vs "south ")
                # OR similar-but-not-identical after normalizing -> likely typo/variant
                is_exact_after_norm = norm1 == norm2
                ratio = SequenceMatcher(None, norm1, norm2).ratio()
                if is_exact_after_norm or ratio >= similarity_threshold:
                    group.append(v2)
                    seen.add(v2)
            if len(group) > 1:
                clusters.append(group)

        if clusters:
            result[col] = clusters

    return result


def run_full_profile(df: pd.DataFrame) -> dict:
    """Runs all profiling checks and returns a combined report."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": profile_missing_values(df),
        "duplicates": profile_duplicates(df),
        "data_types": profile_data_types(df),
        "basic_stats": profile_basic_stats(df),
        "inconsistent_categories": profile_inconsistent_categories(df),
    }
