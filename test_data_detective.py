"""
test_data_detective.py
Basic test suite covering the core logic in each module.
Run with: pytest test_data_detective.py -v
"""

import pandas as pd
import numpy as np
import pytest

from profiler import profile_missing_values, profile_duplicates, profile_data_types, run_full_profile
from anomaly import detect_outliers_iqr, detect_anomalies_isolation_forest
from segmentation import compute_lift_by_category, generate_key_findings
from drift import calculate_psi, detect_drift
from health_score import compute_health_score


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(1, 101),
        "value": [10, 20, 30, 1000] + list(np.random.normal(25, 5, 96)),  # one clear outlier
        "region": ["North"] * 50 + ["South"] * 50,
        "income": [50000] * 90 + [None] * 10,
    })


def test_missing_values_detected(sample_df):
    result = profile_missing_values(sample_df)
    assert result["income"]["missing_count"] == 10
    assert result["income"]["missing_pct"] == 10.0


def test_no_missing_values_in_complete_column(sample_df):
    result = profile_missing_values(sample_df)
    assert result["region"]["missing_count"] == 0


def test_duplicates_detected():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    result = profile_duplicates(df)
    assert result["duplicate_rows"] == 1


def test_data_type_mismatch_detected():
    df = pd.DataFrame({"price": ["10.5", "20.1", "30.0", "40.2"]})
    result = profile_data_types(df)
    assert result["price"]["type_mismatch"] is True
    assert "numeric" in result["price"]["inferred_type"]


def test_outlier_detection_finds_extreme_value(sample_df):
    result = detect_outliers_iqr(sample_df)
    assert "value" in result
    assert 3 in result["value"]["outlier_indices"]  # index of the 1000 value


def test_isolation_forest_runs_without_error(sample_df):
    result = detect_anomalies_isolation_forest(sample_df, contamination=0.05)
    assert "anomaly_count" in result
    assert result["anomaly_count"] >= 0


def test_psi_detects_no_shift_for_identical_distributions():
    np.random.seed(0)
    baseline = pd.Series(np.random.normal(50, 10, 500))
    current = pd.Series(np.random.normal(50, 10, 500))
    psi = calculate_psi(baseline, current)
    assert psi is not None
    assert psi < 0.25  # should not flag as major shift


def test_psi_detects_real_shift():
    np.random.seed(0)
    baseline = pd.Series(np.random.normal(50, 10, 500))
    current = pd.Series(np.random.normal(90, 10, 500))  # clearly shifted
    psi = calculate_psi(baseline, current)
    assert psi is not None
    assert psi > 0.25  # should flag as major shift


def test_lift_finds_overrepresented_category():
    df = pd.DataFrame({
        "region": ["South"] * 20 + ["North"] * 80,
        "value": list(range(100)),
    })
    # anomalies concentrated entirely in South
    anomaly_indices = list(range(20))
    result = compute_lift_by_category(df, anomaly_indices)
    south_lift = next(c["lift"] for c in result["region"]["categories"] if c["category"] == "South")
    assert south_lift > 1.5  # South should be over-represented among anomalies


def test_health_score_within_valid_range(sample_df):
    report = run_full_profile(sample_df)
    health = compute_health_score(sample_df, report)
    assert 0 <= health["health_score"] <= 100


def test_health_score_penalizes_missing_data():
    clean_df = pd.DataFrame({"a": range(100), "b": ["x"] * 100})
    dirty_df = pd.DataFrame({"a": [None] * 50 + list(range(50)), "b": ["x"] * 100})

    clean_report = run_full_profile(clean_df)
    dirty_report = run_full_profile(dirty_df)

    clean_health = compute_health_score(clean_df, clean_report)
    dirty_health = compute_health_score(dirty_df, dirty_report)

    assert clean_health["health_score"] > dirty_health["health_score"]
