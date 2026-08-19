"""
health_score.py
Combines profiling results into a single, explainable 0-100 health score.
"""

import pandas as pd
from profiler import _get_categorical_columns


def compute_health_score(df: pd.DataFrame, profile_report: dict, weights: dict = None) -> dict:
    """
    Computes a weighted composite health score from four sub-scores:
      - Completeness: how much data is present (inverse of missing %)
      - Uniqueness: how much data is non-duplicated
      - Validity: how much data matches its expected/inferred type
      - Consistency: how much categorical data is free of near-duplicate labels

    Default weights sum to 1.0 and can be overridden by the caller.
    """
    if weights is None:
        weights = {"completeness": 0.3, "validity": 0.3, "uniqueness": 0.2, "consistency": 0.2}

    total_rows = len(df)
    total_cols = len(df.columns)

    # --- Completeness ---
    missing_data = profile_report["missing_values"]
    avg_missing_pct = sum(v["missing_pct"] for v in missing_data.values()) / total_cols if total_cols else 0
    completeness = max(0.0, 1 - (avg_missing_pct / 100))

    # --- Uniqueness ---
    dup_pct = profile_report["duplicates"]["duplicate_pct"]
    uniqueness = max(0.0, 1 - (dup_pct / 100))

    # --- Validity (based on type mismatches) ---
    type_data = profile_report["data_types"]
    mismatch_count = sum(1 for v in type_data.values() if v["type_mismatch"])
    validity = max(0.0, 1 - (mismatch_count / total_cols)) if total_cols else 1.0

    # --- Consistency (based on inconsistent category clusters found) ---
    inconsistent = profile_report["inconsistent_categories"]
    categorical_col_count = len(_get_categorical_columns(df)) or 1
    flagged_cols = len(inconsistent)
    consistency = max(0.0, 1 - (flagged_cols / categorical_col_count))

    composite = (
        weights["completeness"] * completeness
        + weights["validity"] * validity
        + weights["uniqueness"] * uniqueness
        + weights["consistency"] * consistency
    )

    score = round(composite * 100)

    return {
        "health_score": score,
        "sub_scores": {
            "completeness": round(completeness * 100, 1),
            "uniqueness": round(uniqueness * 100, 1),
            "validity": round(validity * 100, 1),
            "consistency": round(consistency * 100, 1),
        },
        "weights_used": weights,
    }
