# anomaly_detector.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# =========================
# 1. Volume Spike Detection
# =========================
def detect_volume_spikes(df, window=30, z_threshold=3):
    df = df.copy()

    df["volume_mean"] = (
        df["QUANTITE_NEGOCIEE"]
        .rolling(window)
        .mean()
    )
    df["volume_std"] = (
        df["QUANTITE_NEGOCIEE"]
        .rolling(window)
        .std()
    )

    df["volume_zscore"] = (
        df["QUANTITE_NEGOCIEE"] - df["volume_mean"]
    ) / df["volume_std"]

    df["volume_anomaly"] = df["volume_zscore"].abs() > z_threshold

    return df


# =========================
# 2. Price Variation Detection
# =========================
def detect_price_anomalies(df, threshold=0.05):
    df = df.copy()

    df["price_1h_ago"] = df["CLOTURE"].shift(60)
    df["price_change_1h"] = (
        df["CLOTURE"] - df["price_1h_ago"]
    ) / df["price_1h_ago"]

    df["price_anomaly"] = df["price_change_1h"].abs() > threshold

    return df


# =========================
# 3. Pattern Detection (ML)
# =========================
def detect_pattern_anomalies(df):
    df = df.copy()

    features = df[
        [
            "return_1min",
            "volume_zscore",
            "volatility_rolling",
        ]
    ].dropna()

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )

    model.fit(features)

    df.loc[features.index, "pattern_score"] = model.decision_function(features)
    df.loc[features.index, "pattern_anomaly"] = model.predict(features) == -1

    return df, model


# =========================
# 4. Global Detection (par CODE)
# =========================
def run_anomaly_detection(df):
    df = df.copy()
    df["SEANCE"] = pd.to_datetime(df["SEANCE"])
    df = df.sort_values(["CODE", "SEANCE"])

    results = []
    models = {}

    for code, g in df.groupby("CODE"):
        g = g.copy()

        g["return_1min"] = g["CLOTURE"].pct_change()
        g["volatility_rolling"] = g["return_1min"].rolling(30).std()

        g = detect_volume_spikes(g)
        g = detect_price_anomalies(g)
        g, model = detect_pattern_anomalies(g)

        g["is_anomaly"] = (
            g["volume_anomaly"]
            | g["price_anomaly"]
            | g["pattern_anomaly"]
        )

        results.append(g)
        models[code] = model

    return pd.concat(results), models
