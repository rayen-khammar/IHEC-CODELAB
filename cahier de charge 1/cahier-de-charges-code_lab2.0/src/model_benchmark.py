from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


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
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy().sort_values("DATE").reset_index(drop=True)

    df["LOG_RETURN"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    df["ROLLING_VOL"] = df["LOG_RETURN"].rolling(config.volatility_window).std()
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change(1)

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

    modeling_frame = df[feature_cols + ["TARGET"]]
    modeling_frame = modeling_frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return modeling_frame, feature_cols


def train_test_split_time(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def run_gradient_boosting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, float, float]:
    X_train = train_df.drop(columns=["TARGET"]).to_numpy()
    y_train = train_df["TARGET"].to_numpy()
    X_test = test_df.drop(columns=["TARGET"]).to_numpy()
    y_test = test_df["TARGET"].to_numpy()

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse, mae = evaluate_predictions(y_test, preds)
    return preds, rmse, mae


def run_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str | None = None,
) -> Tuple[np.ndarray, float, float] | None:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("XGBoost not installed. Skipping.")
        return None

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
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model.save_model(os.path.join(model_dir, "xgboost.json"))
    preds = model.predict(X_test)
    rmse, mae = evaluate_predictions(y_test, preds)
    return preds, rmse, mae


def run_arima(
    train_series: pd.Series,
    test_series: pd.Series,
) -> Tuple[np.ndarray, float, float] | None:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("statsmodels not installed. Skipping ARIMA.")
        return None

    model = ARIMA(train_series, order=(5, 1, 0))
    fitted = model.fit()
    preds = fitted.forecast(steps=len(test_series))
    rmse, mae = evaluate_predictions(test_series.to_numpy(), preds.to_numpy())
    return preds.to_numpy(), rmse, mae


def run_sarima(
    train_series: pd.Series,
    test_series: pd.Series,
) -> Tuple[np.ndarray, float, float] | None:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("statsmodels not installed. Skipping SARIMA.")
        return None

    model = SARIMAX(train_series, order=(2, 1, 2), seasonal_order=(1, 0, 1, 5))
    fitted = model.fit(disp=False)
    preds = fitted.forecast(steps=len(test_series))
    rmse, mae = evaluate_predictions(test_series.to_numpy(), preds.to_numpy())
    return preds.to_numpy(), rmse, mae


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    for idx in range(seq_len, len(X)):
        X_seq.append(X[idx - seq_len: idx])
        y_seq.append(y[idx])
    return np.asarray(X_seq), np.asarray(y_seq)


def run_lstm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 10,
    model_dir: str | None = None,
) -> Tuple[np.ndarray, float, float] | None:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch not installed. Skipping LSTM.")
        return None

    X_train = train_df.drop(columns=["TARGET"]).to_numpy()
    y_train = train_df["TARGET"].to_numpy()
    X_test = test_df.drop(columns=["TARGET"]).to_numpy()
    y_test = test_df["TARGET"].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_seq, y_train_seq = _build_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = _build_sequences(X_test, y_test, seq_len)

    if len(X_test_seq) == 0:
        return None

    device = torch.device("cpu")
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32, device=device)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            last = output[:, -1, :]
            return self.fc(last).squeeze(-1)

    model = LSTMRegressor(input_size=X_train_seq.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(8):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    rmse, mae = evaluate_predictions(y_test_seq, preds)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "lstm_state.pt"))
    return preds, rmse, mae


def run_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 10,
) -> Tuple[np.ndarray, float, float] | None:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch not installed. Skipping Transformer.")
        return None

    X_train = train_df.drop(columns=["TARGET"]).to_numpy()
    y_train = train_df["TARGET"].to_numpy()
    X_test = test_df.drop(columns=["TARGET"]).to_numpy()
    y_test = test_df["TARGET"].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_seq, y_train_seq = _build_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = _build_sequences(X_test, y_test, seq_len)

    if len(X_test_seq) == 0:
        return None

    device = torch.device("cpu")
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32, device=device)

    class TransformerRegressor(nn.Module):
        def __init__(self, input_size: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.encoder(x)
            last = x[:, -1, :]
            return self.fc(last).squeeze(-1)

    model = TransformerRegressor(input_size=X_train_seq.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(8):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    rmse, mae = evaluate_predictions(y_test_seq, preds)
    return preds, rmse, mae


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(workspace_root, "multi_company_features.csv")

    companies = ["BIAT", "SFBT", "ATTIJARI BANK"]
    target_col = "TARGET_NEXT_CLOSE"
    test_size = 200
    ensemble_weights = {"XGBoost": 0.6, "GradientBoosting": 0.4}
    run_only_xgboost = False
    run_only_gb_xgb = True

    df = pd.read_csv(features_path)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()

    for company_name in companies:
        subset = df[df["COMPANY"].str.upper() == company_name.upper()].copy()
        if subset.empty:
            raise ValueError(f"Company '{company_name}' not found in multi_company_features.csv")

        subset = subset.sort_values("DATE").reset_index(drop=True)

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
            f"CLOSE_LAG_{lag}" for lag in range(1, 6)
        ] + [
            f"OPEN_LAG_{lag}" for lag in range(1, 6)
        ] + [
            f"HIGH_LAG_{lag}" for lag in range(1, 6)
        ] + [
            f"LOW_LAG_{lag}" for lag in range(1, 6)
        ] + [
            f"VOLUME_LAG_{lag}" for lag in range(1, 6)
        ]

        modeling_data = subset[["DATE"] + feature_cols + [target_col]].dropna().reset_index(drop=True)
        modeling_data = modeling_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        modeling_data = modeling_data.rename(columns={target_col: "TARGET"})

        if len(modeling_data) <= test_size:
            raise ValueError(f"Not enough rows to reserve last {test_size} days as test set.")

        train_df = modeling_data.iloc[:-test_size].copy()
        test_df = modeling_data.iloc[-test_size:].copy()

        train_dates = train_df["DATE"].reset_index(drop=True)
        test_dates = test_df["DATE"].reset_index(drop=True)

        train_model = train_df.drop(columns=["DATE"])
        test_model = test_df.drop(columns=["DATE"])

        print(f"\nCompany: {company_name}")
        print(f"Target : {target_col}")
        print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

        predictions = {}

        model_dir = os.path.join(workspace_root, "models", company_name)

        preds, rmse, mae = run_gradient_boosting(train_model, test_model)
        predictions["GradientBoosting"] = preds
        print(f"GradientBoosting -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        xgb_metrics = run_xgboost(train_model, test_model, model_dir=model_dir)
        if xgb_metrics:
            preds, rmse, mae = xgb_metrics
            predictions["XGBoost"] = preds
            print(f"XGBoost -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        if run_only_xgboost:
            output_dir = os.path.join(workspace_root, "outputs")
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, test_model["TARGET"], label="Actual", color="#000000")
            if "XGBoost" in predictions:
                plt.plot(test_dates, predictions["XGBoost"], label="XGBoost", alpha=0.85)
            plt.title(f"{company_name} - XGBoost Predictions (last {test_size} days)")
            plt.xlabel("Date")
            plt.ylabel("Target")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{company_name}_last_{test_size}_xgboost.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"Saved XGBoost plot to: {plot_path}")
            continue

        if run_only_gb_xgb:
            output_dir = os.path.join(workspace_root, "outputs")
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, test_model["TARGET"], label="Actual", color="#000000")
            if "GradientBoosting" in predictions:
                plt.plot(test_dates, predictions["GradientBoosting"], label="GradientBoosting", alpha=0.85)
            if "XGBoost" in predictions:
                plt.plot(test_dates, predictions["XGBoost"], label="XGBoost", alpha=0.85)

            if all(name in predictions for name in ensemble_weights):
                ensemble = np.zeros_like(predictions["XGBoost"], dtype=float)
                total_weight = 0.0
                for name, weight in ensemble_weights.items():
                    preds = predictions[name]
                    if len(preds) != len(test_dates):
                        offset = len(test_dates) - len(preds)
                        preds = np.concatenate([np.full(offset, np.nan), preds])
                    ensemble = np.nan_to_num(ensemble) + weight * preds
                    total_weight += weight
                if total_weight > 0:
                    ensemble = ensemble / total_weight
                    plt.plot(test_dates, ensemble, label="Ensemble (XGB+GB)", color="#2ca02c", linewidth=2.0)

            plt.title(f"{company_name} - XGBoost vs GradientBoosting (last {test_size} days)")
            plt.xlabel("Date")
            plt.ylabel("Target")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{company_name}_last_{test_size}_xgb_gb.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"Saved comparison plot to: {plot_path}")
            continue

        train_series = train_model["TARGET"]
        test_series = test_model["TARGET"]

        arima_metrics = run_arima(train_series, test_series)
        if arima_metrics:
            preds, rmse, mae = arima_metrics
            predictions["ARIMA"] = preds
            print(f"ARIMA -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        sarima_metrics = run_sarima(train_series, test_series)
        if sarima_metrics:
            preds, rmse, mae = sarima_metrics
            predictions["SARIMA"] = preds
            print(f"SARIMA -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        lstm_metrics = run_lstm(train_model, test_model, model_dir=model_dir)
        if lstm_metrics:
            preds, rmse, mae = lstm_metrics
            predictions["LSTM"] = preds
            print(f"LSTM -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        transformer_metrics = run_transformer(train_model, test_model)
        if transformer_metrics:
            preds, rmse, mae = transformer_metrics
            predictions["Transformer"] = preds
            print(f"Transformer -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        output_dir = os.path.join(workspace_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, test_model["TARGET"], label="Actual", color="#000000")

        for name, preds in predictions.items():
            if len(preds) == len(test_dates):
                plt.plot(test_dates, preds, label=name, alpha=0.85)
            else:
                offset = len(test_dates) - len(preds)
                plt.plot(test_dates.iloc[offset:], preds, label=f"{name} (aligned)", alpha=0.85)

        if all(name in predictions for name in ensemble_weights):
            ensemble = np.zeros_like(predictions["XGBoost"], dtype=float)
            total_weight = 0.0
            for name, weight in ensemble_weights.items():
                preds = predictions[name]
                if len(preds) != len(test_dates):
                    offset = len(test_dates) - len(preds)
                    preds = np.concatenate([np.full(offset, np.nan), preds])
                ensemble = np.nan_to_num(ensemble) + weight * preds
                total_weight += weight
            if total_weight > 0:
                ensemble = ensemble / total_weight
                plt.plot(test_dates, ensemble, label="Ensemble (XGB+LSTM)", color="#2ca02c", linewidth=2.0)

        plt.title(f"{company_name} - Model Predictions (last {test_size} days)")
        plt.xlabel("Date")
        plt.ylabel("Target")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{company_name}_last_{test_size}_predictions.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved prediction plot to: {plot_path}")


if __name__ == "__main__":
    main()
