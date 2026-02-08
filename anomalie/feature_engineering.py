# feature_engineering.py

import pandas as pd
import numpy as np


def compute_features(df, window_volatility=30):
    """
    Create features needed for anomaly detection.
    Input:
        df : DataFrame with at least
             SEANCE, CODE, CLOTURE, QUANTITE ss_NEGOCIEE
    Output:
        df with engineered features
    """

    df = df.copy()

    # Ensure correct time format
    df["SEANCE"] = pd.to_datetime(df["SEANCE"])

    # Sort data by asset and time (CRITICAL for time series)
    df = df.sort_values(["CODE", "SEANCE"])

    # =========================
    # 1. Return (price variation)
    # =========================
    # Percentage change between current close and previous close
    # Used to detect abnormal price movements
    df["return_1min"] = df.groupby("CODE")["CLOTURE"].pct_change()

    # =========================
    # 2. Rolling Volatility
    # =========================
    # Standard deviation of returns over a rolling window
    # Measures how unstable the price is
    df["volatility_rolling"] = (
        df.groupby("CODE")["return_1min"]
        .rolling(window_volatility)
        .std()
        .reset_index(level=0, drop=True)
    )

    # =========================
    # 3. Log Volume (optional but useful)
    # =========================
    # Stabilizes extreme volume values
    df["log_volume"] = np.log1p(df["QUANTITE_NEGOCIEE"])

    # =========================
    # 4. Rolling Average Volume
    # =========================
    # Used later to compare current volume to normal behavior
    df["volume_rolling_mean"] = (
        df.groupby("CODE")["QUANTITE_NEGOCIEE"]
        .rolling(30)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df
