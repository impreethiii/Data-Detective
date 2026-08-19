"""
anomaly.py
Outlier/anomaly detection: IQR (per-column, univariate) and Isolation Forest (multivariate).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    Per-column outlier detection using the IQR method.
    Returns flagged row indices per numeric column.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
        outlier_indices = df[outlier_mask.fillna(False)].index.tolist()

        if outlier_indices:
            result[col] = {
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "outlier_count": len(outlier_indices),
                "outlier_indices": outlier_indices,
            }

    return result


def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """
    Multivariate anomaly detection using Isolation Forest across all numeric columns
    at once. Flags rows that are jointly unusual, even if no single column looks
    extreme on its own.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if numeric_df.shape[0] < 10 or numeric_df.shape[1] < 1:
        return {"anomaly_indices": [], "note": "Not enough numeric data to run Isolation Forest."}

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(numeric_df)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(numeric_df)  # lower = more anomalous

    anomaly_mask = predictions == -1
    anomaly_indices = numeric_df.index[anomaly_mask].tolist()

    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_count": len(anomaly_indices),
        "anomaly_pct": round(len(anomaly_indices) / len(numeric_df) * 100, 2),
    }


def get_combined_anomaly_indices(df: pd.DataFrame, contamination: float = 0.05) -> list:
    """
    Combines IQR-based and Isolation-Forest-based anomalies into one set of row
    indices, used downstream by the root-cause/segmentation engine.
    """
    iqr_results = detect_outliers_iqr(df)
    iso_results = detect_anomalies_isolation_forest(df, contamination=contamination)

    combined = set(iso_results.get("anomaly_indices", []))
    for col_result in iqr_results.values():
        combined.update(col_result["outlier_indices"])

    return sorted(combined)
