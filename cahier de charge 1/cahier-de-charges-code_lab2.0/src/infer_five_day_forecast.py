from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    n_lags: int = 5
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


def load_bvmt_files(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.lower().endswith(".csv"):
            df = _load_csv(path)
        else:
            df = _load_txt(path)
        df["SOURCE_FILE"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise ValueError("No input files found.")

    return pd.concat(frames, ignore_index=True)


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
    df = df.loc[:, ~df.columns.duplicated()]

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


def drop_non_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_zero = (df["OPEN"] == 0) & (df["HIGH"] == 0) & (df["LOW"] == 0) & (df["CLOSE"] == 0)
    volume_zero = df["VOLUME"] == 0
    return df[~(price_zero & volume_zero)].copy()


def add_features(df: pd.DataFrame, config: FeatureConfig) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy().sort_values("DATE").reset_index(drop=True)

    prev_close = df["CLOSE"].shift(1)
    valid = (df["CLOSE"] > 0) & (prev_close > 0)
    df["LOG_RETURN"] = np.where(valid, np.log(df["CLOSE"] / prev_close), np.nan)
    df["ROLLING_VOL"] = df["LOG_RETURN"].rolling(config.volatility_window).std()
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change(1)
    df["ROLLING_MEAN"] = df["CLOSE"].rolling(config.n_lags).mean()
    df["ROLLING_MEDIAN"] = df["CLOSE"].rolling(config.n_lags).median()
    df["ROLLING_STD"] = df["CLOSE"].rolling(config.n_lags).std()

    for lag in range(1, config.n_lags + 1):
        df[f"CLOSE_LAG_{lag}"] = df["CLOSE"].shift(lag)
        df[f"OPEN_LAG_{lag}"] = df["OPEN"].shift(lag)
        df[f"HIGH_LAG_{lag}"] = df["HIGH"].shift(lag)
        df[f"LOW_LAG_{lag}"] = df["LOW"].shift(lag)
        df[f"VOLUME_LAG_{lag}"] = df["VOLUME"].shift(lag)

    feature_cols = [
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOLUME",
        "LOG_RETURN",
        "ROLLING_VOL",
        "VOLUME_CHANGE",
        "ROLLING_MEAN",
        "ROLLING_MEDIAN",
        "ROLLING_STD",
    ] + [
        f"CLOSE_LAG_{lag}" for lag in range(1, config.n_lags + 1)
    ] + [
        f"OPEN_LAG_{lag}" for lag in range(1, config.n_lags + 1)
    ] + [
        f"HIGH_LAG_{lag}" for lag in range(1, config.n_lags + 1)
    ] + [
        f"LOW_LAG_{lag}" for lag in range(1, config.n_lags + 1)
    ] + [
        f"VOLUME_LAG_{lag}" for lag in range(1, config.n_lags + 1)
    ]

    feature_frame = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return feature_frame, feature_cols


def load_models(model_dir: str):
    models = {}

    xgb_path = os.path.join(model_dir, "xgboost.json")
    if os.path.exists(xgb_path):
        from xgboost import XGBRegressor

        xgb = XGBRegressor()
        xgb.load_model(xgb_path)
        models["XGBoost"] = xgb

    gb_path = os.path.join(model_dir, "gradient_boosting.pkl")
    if os.path.exists(gb_path):
        import joblib

        models["GradientBoosting"] = joblib.load(gb_path)

    if not models:
        raise ValueError(f"No models found in {model_dir}")

    return models


def recursive_forecast(
    models: dict,
    history_df: pd.DataFrame,
    horizon: int,
    config: FeatureConfig,
) -> pd.DataFrame:
    df = history_df.copy().sort_values("DATE").reset_index(drop=True)

    price_zero = (df["OPEN"] == 0) & (df["HIGH"] == 0) & (df["LOW"] == 0) & (df["CLOSE"] == 0)
    volume_zero = df["VOLUME"] == 0
    df = df[~(price_zero & volume_zero)].reset_index(drop=True)

    base_close = float(df["CLOSE"].iloc[-1])

    results = []
    for step in range(1, horizon + 1):
        features, feature_cols = add_features(df, config)
        last_features = features.iloc[[-1]][feature_cols]

        preds = []
        for model in models.values():
            preds.append(float(model.predict(last_features.to_numpy())[0]))

        pred = float(np.mean(preds))

        results.append(
            {
                "step": step,
                "pred_close": pred,
                "pct_change_from_last": (pred / base_close - 1.0) * 100.0,
            }
        )

        next_row = {
            "DATE": df["DATE"].iloc[-1] + pd.Timedelta(days=1),
            "OPEN": pred,
            "HIGH": pred,
            "LOW": pred,
            "CLOSE": pred,
            "VOLUME": df["VOLUME"].iloc[-1],
        }
        df = pd.concat([df, pd.DataFrame([next_row])], ignore_index=True)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="5-day recursive forecast per company.")
    parser.add_argument("--company", required=True, help="Company name (e.g., BIAT)")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Input file paths or glob patterns (defaults to histo_cotation_*.csv/txt)",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date (dd/mm/yyyy). Forecast starts after this date.",
    )

    args = parser.parse_args()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.files:
        expanded = []
        for pattern in args.files:
            expanded.extend(glob.glob(pattern))
        data_paths = expanded
    else:
        data_paths = (
            glob.glob(os.path.join(workspace_root, "histo_cotation_*.csv"))
            + glob.glob(os.path.join(workspace_root, "histo_cotation_*.txt"))
        )

    raw = load_bvmt_files(sorted(data_paths))
    normalized = normalize_market_data(raw)

    company_df = normalized[normalized["COMPANY"].str.upper() == args.company.upper()].copy()
    if company_df.empty:
        raise ValueError(f"Company '{args.company}' not found in data.")

    company_df = company_df.sort_values("DATE").reset_index(drop=True)

    effective_date = company_df["DATE"].max()
    if args.start_date:
        start_date = pd.to_datetime(args.start_date, dayfirst=True, errors="coerce")
        if pd.isna(start_date):
            raise ValueError("Invalid --start-date. Use dd/mm/yyyy format.")
        max_date = company_df["DATE"].max()
        if start_date > max_date:
            raise ValueError(
                f"Start date is beyond available data (max {max_date.strftime('%d/%m/%Y')})."
            )
        company_df = company_df[company_df["DATE"] <= start_date].copy()
        if company_df.empty:
            raise ValueError("No data available on or before the selected start date.")

    company_df = drop_non_trading_days(company_df)
    if company_df.empty:
        raise ValueError("No trading rows available after filtering non-trading days.")

    effective_date = company_df["DATE"].max()

    model_dir = os.path.join(workspace_root, "models", args.company)
    models = load_models(model_dir)

    config = FeatureConfig(n_lags=5, volatility_window=10)
    forecast_df = recursive_forecast(models, company_df, args.horizon, config)

    print(f"Company: {args.company}")
    print(f"Using models: {', '.join(models.keys())}")
    if args.start_date:
        print(f"Requested start date: {args.start_date}")
        print(f"Effective as-of date: {effective_date.strftime('%d/%m/%Y')}")
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()
