from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance


@dataclass
class FeatureConfig:
    n_lags: int = 5
    volatility_window: int = 10
    rolling_window: int = 5


def _standardize_columns(columns: List[str]) -> List[str]:
    normalized = []
    for col in columns:
        clean = col.strip().upper().replace(" ", "_")
        normalized.append(clean)
    return normalized


def load_single_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=";", engine="python")
    else:
        df = pd.read_fwf(path, skiprows=[1])
    df.columns = _standardize_columns(df.columns)
    return df


def normalize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    column_map = {
        "SEANCE": "DATE",
        "LIB_VAL": "COMPANY",
        "CODE_VAL": "CODE",
        "C_GR_RLC": "GROUPE",
        "OUVERTURE": "OPEN",
        "PLUS_HAUT": "HIGH",
        "PLUS_BAS": "LOW",
        "CLOTURE": "CLOSE",
        "QUANTITE_NEGOCIEE": "VOLUME",
        "NB_TRAN": "NB_TRANSACTION",
        "VALEUR": "COMPANY",
        "CODE": "CODE",
    }
    available_map = {key: value for key, value in column_map.items() if key in df.columns}
    df = df.rename(columns=available_map)

    required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "COMPANY", "CODE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    df["CODE"] = df["CODE"].astype(str).str.strip()

    df = df.dropna(subset=["DATE", "CLOSE"])
    df = df.sort_values(["COMPANY", "DATE"]).reset_index(drop=True)
    return df


def list_companies(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    counts = df.groupby(["COMPANY", "CODE"]).size().reset_index(name="ROWS")
    return counts.sort_values("ROWS", ascending=False).head(top_n)


def filter_company(df: pd.DataFrame, company: str) -> pd.DataFrame:
    subset = df[df["COMPANY"].str.upper() == company.upper()].copy()
    if subset.empty:
        raise ValueError(f"Company '{company}' not found.")
    return subset.sort_values("DATE").reset_index(drop=True)


def check_open_vs_prev_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PREV_CLOSE"] = df["CLOSE"].shift(1)
    df["OPEN_MINUS_PREV_CLOSE"] = df["OPEN"] - df["PREV_CLOSE"]
    return df


def add_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = df.copy().sort_values("DATE").reset_index(drop=True)

    df["LOG_RETURN"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    df["ROLLING_VOL"] = df["LOG_RETURN"].rolling(config.volatility_window).std()
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change(1)

    df["ROLLING_MEAN_5"] = df["CLOSE"].rolling(config.rolling_window).mean()
    df["ROLLING_STD_5"] = df["CLOSE"].rolling(config.rolling_window).std()

    for lag in range(1, config.n_lags + 1):
        df[f"CLOSE_LAG_{lag}"] = df["CLOSE"].shift(lag)

    df["TARGET"] = df["CLOSE"].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def summarize_metrics(df: pd.DataFrame) -> None:
    print("\nSummary metrics:")
    metrics = {
        "rows": len(df),
        "date_min": df["DATE"].min(),
        "date_max": df["DATE"].max(),
        "close_mean": df["CLOSE"].mean(),
        "close_std": df["CLOSE"].std(),
        "volume_mean": df["VOLUME"].mean(),
        "volume_std": df["VOLUME"].std(),
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")


def plot_basic_charts(df: pd.DataFrame, company: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["DATE"], df["CLOSE"], label="Close")
    plt.title(f"{company} - Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.bar(df["DATE"], df["VOLUME"], color="#4C78A8")
    plt.title(f"{company} - Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(df["DATE"], df["ROLLING_VOL"], color="#F58518")
    plt.title(f"{company} - Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.show()


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).dropna()
    corr = numeric.corr()[["TARGET"]].sort_values("TARGET", ascending=False)
    return corr


def model_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOLUME",
        "LOG_RETURN",
        "ROLLING_VOL",
        "VOLUME_CHANGE",
        "ROLLING_MEAN_5",
        "ROLLING_STD_5",
    ] + [col for col in df.columns if col.startswith("CLOSE_LAG_")]

    modeling_df = df[feature_cols + ["TARGET"]].dropna()
    X = modeling_df[feature_cols].to_numpy()
    y = modeling_df["TARGET"].to_numpy()

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    perm = permutation_importance(model, X, y, n_repeats=15, random_state=42)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(workspace_root, "histo_cotation_2022.csv")

    raw = load_single_file(file_path)
    normalized = normalize_market_data(raw)

    print("Top companies in file:")
    print(list_companies(normalized, top_n=10).to_string(index=False))

    company = "BIAT"
    company_df = filter_company(normalized, company)

    print(f"\nSelected company: {company}")
    summarize_metrics(company_df)

    company_df = check_open_vs_prev_close(company_df)
    mismatch = company_df["OPEN_MINUS_PREV_CLOSE"].abs().dropna()
    print("\nOpen vs previous close:")
    print(mismatch.describe())

    feature_cfg = FeatureConfig(n_lags=5, volatility_window=10, rolling_window=5)
    company_features = add_features(company_df, feature_cfg)

    plot_basic_charts(company_features, company)

    corr = correlation_with_target(company_features)
    print("\nCorrelation with TARGET (next close):")
    print(corr.to_string())

    importance = model_feature_importance(company_features)
    print("\nPermutation importance (GradientBoostingRegressor):")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
