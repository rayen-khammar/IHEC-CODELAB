from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class FeatureConfig:
    n_lags: int = 5
    volatility_window: int = 10
    volume_change_window: int = 1


@dataclass
class WalkForwardConfig:
    train_window: int = 252 * 2  # ~2 years of trading days
    test_window: int = 20
    step: int = 20


def _standardize_columns(columns: Iterable[str]) -> List[str]:
    normalized = []
    for col in columns:
        clean = col.strip().upper().replace(" ", "_")
        normalized.append(clean)
    return normalized


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

    full = pd.concat(frames, ignore_index=True)
    return full


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


def list_companies(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    counts = df.groupby(["COMPANY", "CODE"]).size().reset_index(name="ROWS")
    return counts.sort_values("ROWS", ascending=False).head(top_n)


def filter_company(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    company_mask = df["COMPANY"].str.upper() == company_name.upper()
    filtered = df.loc[company_mask].copy()
    if filtered.empty:
        raise ValueError(f"Company '{company_name}' not found.")
    return filtered.sort_values("DATE").reset_index(drop=True)


def add_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().sort_values("DATE").reset_index(drop=True)

    df["LOG_RETURN"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    df["ROLLING_VOL"] = (
        df["LOG_RETURN"].rolling(config.volatility_window).std()
    )
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change(config.volume_change_window)

    for lag in range(1, config.n_lags + 1):
        df[f"CLOSE_LAG_{lag}"] = df["CLOSE"].shift(lag)

    feature_cols = [
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOLUME",
        "LOG_RETURN",
        "ROLLING_VOL",
        "VOLUME_CHANGE",
    ] + [f"CLOSE_LAG_{lag}" for lag in range(1, config.n_lags + 1)]

    df["TARGET"] = df[target_col].shift(-1)

    feature_frame = df[feature_cols]
    modeling_frame = df[feature_cols + ["TARGET"]]
    modeling_frame = modeling_frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return modeling_frame, feature_frame


def walk_forward_validation(
    model: GradientBoostingRegressor,
    data: pd.DataFrame,
    config: WalkForwardConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=["TARGET"]).to_numpy()
    y = data["TARGET"].to_numpy()

    predictions = []
    targets = []

    n_samples = len(data)
    train_end = config.train_window

    while train_end + config.test_window <= n_samples:
        test_end = train_end + config.test_window
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions.append(y_pred)
        targets.append(y_test)

        train_end += config.step

    if not predictions:
        raise ValueError("Not enough data for walk-forward validation. Reduce window sizes.")

    return np.concatenate(predictions), np.concatenate(targets)


def train_final_model(
    model: GradientBoostingRegressor,
    data: pd.DataFrame,
) -> GradientBoostingRegressor:
    X = data.drop(columns=["TARGET"]).to_numpy()
    y = data["TARGET"].to_numpy()
    model.fit(X, y)
    return model


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = sorted(
        glob.glob(os.path.join(workspace_root, "histo_cotation_*.csv"))
        + glob.glob(os.path.join(workspace_root, "histo_cotation_*.txt"))
    )

    raw = load_bvmt_files(data_paths)
    normalized = normalize_market_data(raw)

    print("Top companies by available rows:")
    print(list_companies(normalized, top_n=15).to_string(index=False))

    company_name = "BIAT"
    company_df = filter_company(normalized, company_name)

    feature_cfg = FeatureConfig(n_lags=5, volatility_window=10)
    target_fields = ["CLOSE", "HIGH", "LOW"]

    wf_cfg = WalkForwardConfig(train_window=252 * 2, test_window=20, step=20)

    for target in target_fields:
        modeling_data, full_features = add_features(company_df, feature_cfg, target)
        model = GradientBoostingRegressor(random_state=42)

        preds, actuals = walk_forward_validation(model, modeling_data, wf_cfg)

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)

        print(f"\nWalk-forward validation metrics ({target}):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")

        final_model = train_final_model(model, modeling_data)

        last_row = full_features.iloc[[-1]].to_numpy()
        next_day_pred = final_model.predict(last_row)[0]

        last_date = company_df["DATE"].iloc[-1].date()
        print(f"\nNext-day prediction ({target}):")
        print(f"Company: {company_name}")
        print(f"Last date: {last_date}")
        print(f"Predicted next {target.lower()}: {next_day_pred:.4f}")


if __name__ == "__main__":
    main()
