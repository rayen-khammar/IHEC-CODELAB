from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def add_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
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

    feature_cols = (
        [
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
        ]
        + [f"CLOSE_LAG_{lag}" for lag in range(1, config.n_lags + 1)]
        + [f"OPEN_LAG_{lag}" for lag in range(1, config.n_lags + 1)]
        + [f"HIGH_LAG_{lag}" for lag in range(1, config.n_lags + 1)]
        + [f"LOW_LAG_{lag}" for lag in range(1, config.n_lags + 1)]
        + [f"VOLUME_LAG_{lag}" for lag in range(1, config.n_lags + 1)]
    )

    df["TARGET"] = df["CLOSE"].shift(-1)

    modeling_frame = df[feature_cols + ["TARGET"]]
    modeling_frame = modeling_frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return modeling_frame


def train_test_split_last(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_size:
        raise ValueError("Not enough rows to reserve test set.")
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def run_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, model_dir: str) -> tuple[float, float]:
    from xgboost import XGBRegressor

    X_train = train_df.drop(columns=["TARGET"]).to_numpy()
    y_train = train_df["TARGET"].to_numpy()
    X_test = test_df.drop(columns=["TARGET"]).to_numpy()
    y_test = test_df["TARGET"].to_numpy()

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "xgboost.json"))

    return evaluate_predictions(y_test, preds)


def run_gradient_boosting(train_df: pd.DataFrame, test_df: pd.DataFrame, model_dir: str) -> tuple[float, float]:
    import joblib

    X_train = train_df.drop(columns=["TARGET"]).to_numpy()
    y_train = train_df["TARGET"].to_numpy()
    X_test = test_df.drop(columns=["TARGET"]).to_numpy()
    y_test = test_df["TARGET"].to_numpy()

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "gradient_boosting.pkl"))

    return evaluate_predictions(y_test, preds)


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = sorted(
        glob.glob(os.path.join(workspace_root, "histo_cotation_*.csv"))
        + glob.glob(os.path.join(workspace_root, "histo_cotation_*.txt"))
    )

    raw = load_bvmt_files(data_paths)
    normalized = normalize_market_data(raw)

    config = FeatureConfig(n_lags=5, volatility_window=10)
    test_size = 200

    metrics = []
    companies = sorted(normalized["COMPANY"].dropna().unique().tolist())

    for company in companies:
        company_df = normalized[normalized["COMPANY"] == company].copy()
        if len(company_df) < (test_size + config.n_lags + config.volatility_window + 2):
            continue

        modeling_data = add_features(company_df, config)
        if len(modeling_data) <= test_size:
            continue

        train_df, test_df = train_test_split_last(modeling_data, test_size)

        model_dir = os.path.join(workspace_root, "models", company)

        try:
            xgb_rmse, xgb_mae = run_xgboost(train_df, test_df, model_dir)
            gb_rmse, gb_mae = run_gradient_boosting(train_df, test_df, model_dir)
        except Exception as exc:
            print(f"Skipping {company}: {exc}")
            continue

        metrics.append(
            {
                "COMPANY": company,
                "ROWS": len(modeling_data),
                "XGBOOST_RMSE": xgb_rmse,
                "XGBOOST_MAE": xgb_mae,
                "GBOOST_RMSE": gb_rmse,
                "GBOOST_MAE": gb_mae,
            }
        )

        print(f"{company}: XGB RMSE={xgb_rmse:.4f}, GBoost RMSE={gb_rmse:.4f}")

    if metrics:
        output_path = os.path.join(workspace_root, "outputs", "all_company_metrics.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(metrics).to_csv(output_path, index=False)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
