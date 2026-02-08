# evaluation.py

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# =========================
# 1. Label Preparation
# =========================
def prepare_labels(df_detected, df_ground_truth, anomaly_column=None):
    """
    Build y_true / y_pred vectors
    If anomaly_column is provided, evaluate a specific anomaly type
    """

    df_detected = df_detected.copy()
    df_ground_truth = df_ground_truth.copy()

    df_detected["SEANCE"] = pd.to_datetime(df_detected["SEANCE"])
    df_ground_truth["SEANCE"] = pd.to_datetime(df_ground_truth["SEANCE"])

    # Build matching key
    df_detected["key"] = (
        df_detected["CODE"].astype(str)
        + "_"
        + df_detected["SEANCE"].astype(str)
    )

    df_ground_truth["key"] = (
        df_ground_truth["CODE"].astype(str)
        + "_"
        + df_ground_truth["SEANCE"].astype(str)
    )

    # Ground truth
    if anomaly_column:
        gt_keys = df_ground_truth[
            df_ground_truth["anomaly_type"] == anomaly_column
        ]["key"]
    else:
        gt_keys = df_ground_truth["key"]

    y_true = df_detected["key"].isin(gt_keys).astype(int)

    # Prediction
    if anomaly_column:
        y_pred = df_detected[f"{anomaly_column}_anomaly"].astype(int)
    else:
        y_pred = df_detected["is_anomaly"].astype(int)

    return y_true, y_pred


# =========================
# 2. Metric Computation
# =========================
def compute_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


# =========================
# 3. Full Evaluation
# =========================
def evaluate_detection(df_detected, df_ground_truth):
    """
    Evaluate global performance + per anomaly type
    """

    results = {}

    # Global evaluation
    y_true, y_pred = prepare_labels(df_detected, df_ground_truth)
    results["global"] = compute_metrics(y_true, y_pred)

    # Per-type evaluation
    for anomaly_type in ["volume", "price", "pattern"]:
        y_true, y_pred = prepare_labels(
            df_detected,
            df_ground_truth,
            anomaly_column=anomaly_type
        )
        results[anomaly_type] = compute_metrics(y_true, y_pred)

    return results
