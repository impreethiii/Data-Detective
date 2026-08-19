"""
segmentation.py
The "root cause" engine. Given a set of anomalous row indices, this finds which
categorical segments are over-represented among anomalies compared to their
share of the overall dataset, and expresses that as a plain-English finding.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from profiler import _get_categorical_columns


def compute_lift_by_category(df: pd.DataFrame, anomaly_indices: list, max_cardinality: int = 50) -> dict:
    """
    For each categorical column, computes 'lift' per category value:
        lift = (share of anomalies in this category) / (share of all rows in this category)

    A lift > 1 means that category is over-represented among anomalies.
    Also runs a chi-square test to check whether the anomaly distribution across
    categories is statistically different from what random chance would produce.
    """
    if not anomaly_indices:
        return {}

    categorical_cols = _get_categorical_columns(df)
    total_rows = len(df)
    anomaly_df = df.loc[df.index.isin(anomaly_indices)]
    total_anomalies = len(anomaly_df)

    results = {}

    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique < 2 or n_unique > max_cardinality:
            continue  # skip near-constant or very high-cardinality (likely ID) columns

        overall_counts = df[col].value_counts(dropna=True)
        anomaly_counts = anomaly_df[col].value_counts(dropna=True)

        category_results = []
        for category in overall_counts.index:
            overall_share = overall_counts[category] / total_rows
            anomaly_share = anomaly_counts.get(category, 0) / total_anomalies if total_anomalies > 0 else 0
            lift = round(anomaly_share / overall_share, 2) if overall_share > 0 else 0.0

            category_results.append({
                "category": str(category),
                "overall_pct": round(overall_share * 100, 2),
                "anomaly_pct": round(anomaly_share * 100, 2),
                "lift": lift,
            })

        # sort by lift descending — biggest over-representation first
        category_results.sort(key=lambda x: x["lift"], reverse=True)

        # chi-square test: is the anomaly distribution significantly different from overall?
        try:
            contingency = pd.DataFrame({
                "overall": overall_counts,
                "anomaly": anomaly_counts.reindex(overall_counts.index, fill_value=0),
            }).fillna(0)
            chi2, p_value, _, _ = chi2_contingency(contingency.T)
            significant = bool(p_value < 0.05)
        except Exception:
            p_value = None
            significant = False

        results[col] = {
            "categories": category_results,
            "chi_square_p_value": round(float(p_value), 5) if p_value is not None else None,
            "statistically_significant": significant,
        }

    return results


def generate_key_findings(lift_results: dict, top_n: int = 3) -> list:
    """
    Turns the lift analysis into plain-English findings, e.g.:
    "78% of anomalous transactions come from 2 categories in 'product_category'."
    Only surfaces findings that are statistically significant AND have meaningfully
    high lift, to avoid noisy/false claims.
    """
    findings = []

    for col, data in lift_results.items():
        if not data["statistically_significant"]:
            continue

        # take top categories that meaningfully over-represent (lift > 1.5)
        strong_categories = [c for c in data["categories"] if c["lift"] > 1.5 and c["anomaly_pct"] > 5]
        if not strong_categories:
            continue

        combined_anomaly_share = sum(c["anomaly_pct"] for c in strong_categories[:2])
        category_names = [c["category"] for c in strong_categories[:2]]

        if len(category_names) == 1:
            text = (
                f"{combined_anomaly_share:.0f}% of anomalies are concentrated in "
                f"'{category_names[0]}' within column '{col}' "
                f"(lift: {strong_categories[0]['lift']}x normal rate)."
            )
        else:
            text = (
                f"{combined_anomaly_share:.0f}% of anomalies come from just "
                f"{len(category_names)} values in '{col}': {', '.join(category_names)}."
            )

        findings.append({
            "column": col,
            "text": text,
            "p_value": data["chi_square_p_value"],
        })

    # sort strongest findings first (lowest p-value = most significant)
    findings.sort(key=lambda f: f["p_value"] if f["p_value"] is not None else 1)
    return findings[:top_n]
