from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    n_lags: int = 5
    rolling_window: int = 5
    volatility_window: int = 10


def _standardize_columns(columns: Iterable[str]) -> List[str]:
    return [col.strip().upper().replace(" ", "_") for col in columns]


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")
    df.columns = _standardize_columns(df.columns)
    return df


def _load_txt(path: str) -> pd.DataFrame:
    df = pd.read_fwf(path, skiprows=[1])
    df.columns = _standardize_columns(df.columns)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df.dropna(subset=["DATE", "CLOSE"]).reset_index(drop=True)
    return df


def load_all_market_files(workspace_root: str) -> pd.DataFrame:
    data_paths = sorted(
        glob.glob(os.path.join(workspace_root, "histo_cotation_*.csv"))
        + glob.glob(os.path.join(workspace_root, "histo_cotation_*.txt"))
    )

    frames = []
    for path in data_paths:
        try:
            if path.lower().endswith(".csv"):
                df = _load_csv(path)
            else:
                df = _load_txt(path)
            df = _normalize(df)
            df["SOURCE_FILE"] = os.path.basename(path)
            frames.append(df)
        except ValueError as exc:
            print(f"Skipping {os.path.basename(path)}: {exc}")

    if not frames:
        raise ValueError("No compatible input files found.")

    return pd.concat(frames, ignore_index=True)


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


def drop_non_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_zero = (df["OPEN"] == 0) & (df["HIGH"] == 0) & (df["LOW"] == 0) & (df["CLOSE"] == 0)
    volume_zero = df["VOLUME"] == 0
    df = df[~(price_zero & volume_zero)].copy()
    return df


def add_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = df.copy().sort_values(["COMPANY", "DATE"]).reset_index(drop=True)

    df["LOG_RETURN"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    df["ROLLING_VOL"] = df["LOG_RETURN"].rolling(config.volatility_window).std()
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change(1)

    df["ROLLING_MEAN"] = df["CLOSE"].rolling(config.rolling_window).mean()
    df["ROLLING_MEDIAN"] = df["CLOSE"].rolling(config.rolling_window).median()
    df["ROLLING_STD"] = df["CLOSE"].rolling(config.rolling_window).std()

    for lag in range(1, config.n_lags + 1):
        df[f"CLOSE_LAG_{lag}"] = df["CLOSE"].shift(lag)
        df[f"OPEN_LAG_{lag}"] = df["OPEN"].shift(lag)
        df[f"HIGH_LAG_{lag}"] = df["HIGH"].shift(lag)
        df[f"LOW_LAG_{lag}"] = df["LOW"].shift(lag)
        df[f"VOLUME_LAG_{lag}"] = df["VOLUME"].shift(lag)

    df["TARGET_NEXT_CLOSE"] = df["CLOSE"].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    companies = ["BIAT", "SFBT", "ATTIJARI BANK"]

    full = load_all_market_files(workspace_root)
    full = drop_all_null_columns(full)
    full = drop_non_trading_days(full)

    subset = full[full["COMPANY"].str.upper().isin([c.upper() for c in companies])].copy()
    if subset.empty:
        raise ValueError("No rows found for the selected companies.")

    subset = subset.sort_values(["COMPANY", "DATE"]).reset_index(drop=True)

    feature_cfg = FeatureConfig(n_lags=5, rolling_window=5, volatility_window=10)
    featured = add_features(subset, feature_cfg)

    output_raw = os.path.join(workspace_root, "multi_company_raw.csv")
    output_features = os.path.join(workspace_root, "multi_company_features.csv")

    subset.to_csv(output_raw, index=False)
    featured.to_csv(output_features, index=False)

    print(f"Exported raw subset to {output_raw}")
    print(f"Exported feature dataset to {output_features}")


if __name__ == "__main__":
    main()
