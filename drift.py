"""
drift.py
Distribution drift detection between two datasets (or two time periods of the
same dataset): KS-test (statistical significance) and PSI (magnitude of shift,
industry-standard in finance/risk modeling).
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index. Buckets both distributions into the same bins
    (based on baseline quantiles) and compares bucket proportions.
    Rule of thumb: PSI < 0.1 = no significant shift, 0.1-0.25 = moderate shift,
    > 0.25 = major shift.
    """
    baseline = baseline.dropna()
    current = current.dropna()

    if len(baseline) < 10 or len(current) < 10:
        return None

    # build bin edges from baseline quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(baseline.quantile(quantiles).values)
    if len(bin_edges) < 2:
        return None
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_counts = pd.cut(baseline, bins=bin_edges).value_counts(sort=False)
    current_counts = pd.cut(current, bins=bin_edges).value_counts(sort=False)

    baseline_pct = (baseline_counts / len(baseline)).replace(0, 0.0001)
    current_pct = (current_counts / len(current)).replace(0, 0.0001)

    psi = float(((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)).sum())
    return round(psi, 4)


def detect_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Compares numeric columns shared between two dataframes (e.g. 'last month'
    vs 'this month') and flags columns whose distribution shifted meaningfully.
    """
    shared_numeric_cols = [
        col for col in baseline_df.select_dtypes(include=[np.number]).columns
        if col in current_df.columns
    ]

    results = {}
    for col in shared_numeric_cols:
        baseline_series = baseline_df[col].dropna()
        current_series = current_df[col].dropna()

        if len(baseline_series) < 10 or len(current_series) < 10:
            continue

        ks_stat, p_value = ks_2samp(baseline_series, current_series)
        psi = calculate_psi(baseline_series, current_series)

        drifted = bool(p_value < 0.05) or (psi is not None and psi > 0.25)

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(p_value), 5),
            "psi": psi,
            "baseline_mean": round(float(baseline_series.mean()), 4),
            "current_mean": round(float(current_series.mean()), 4),
            "drift_detected": drifted,
        }

    return results


def split_by_date(df: pd.DataFrame, date_col: str, split_point=None):
    """
    Utility: splits a single dataframe into 'baseline' (earlier) and 'current'
    (later) periods using a date column, for drift comparison without needing
    two separate files. If split_point is None, splits at the median date.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if split_point is None:
        split_point = df[date_col].median()
    else:
        split_point = pd.to_datetime(split_point)

    baseline = df[df[date_col] < split_point]
    current = df[df[date_col] >= split_point]
    return baseline, current, split_point
