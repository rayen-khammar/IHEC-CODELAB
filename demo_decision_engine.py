"""Demo script to test the Decision Engine end-to-end.

It attempts:
- Live-ish BVMT scraping (if reachable)
- Sentiment analysis for a provided symbol/company
- 5-day price prediction using existing models
- Anomaly detection from scraped metrics + model error
- Portfolio projection using predicted prices

All external modules are called defensively; failures fall back to safe defaults.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from decision_engine import DecisionEngine, DecisionConfig, safe_call


logger = logging.getLogger("demo_decision_engine")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def try_scrape_bvmt(url: str) -> Optional[pd.DataFrame]:
    ingestion_path = os.path.join(ROOT, "IHEC-CODELAB", "app", "ingestion.py")
    if not os.path.exists(ingestion_path):
        logger.warning("Ingestion module not found at %s", ingestion_path)
        return None

    app_root = os.path.join(ROOT, "IHEC-CODELAB")
    if app_root not in sys.path:
        sys.path.insert(0, app_root)

    ingestion = load_module(ingestion_path, "ingestion")
    run_one_cycle = getattr(ingestion, "run_one_cycle", None)
    return safe_call(run_one_cycle, url, default=None, name="run_one_cycle")


def get_bvmt_url() -> str:
    config_path = os.path.join(ROOT, "IHEC-CODELAB", "app", "config.py")
    if not os.path.exists(config_path):
        return "https://www.bvmt.com.tn/"
    config = load_module(config_path, "config")
    return getattr(config, "BVMT_URL", "https://www.bvmt.com.tn/")


def select_symbol(df: Optional[pd.DataFrame], symbol: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return symbol, None

    cols = [c.upper() for c in df.columns]
    code_col = "CODE" if "CODE" in cols else None
    val_col = "VALEUR" if "VALEUR" in cols else None

    if symbol and code_col:
        match = df[df[code_col].astype(str).str.upper() == symbol.upper()]
        if not match.empty:
            company = match[val_col].iloc[-1] if val_col else None
            return symbol, company

    if code_col:
        code = str(df[code_col].iloc[0]).strip()
    else:
        code = symbol

    company = None
    if val_col and code_col:
        company = df[val_col].iloc[0]
    return code, company


def latest_price_from_scrape(df: Optional[pd.DataFrame], symbol: Optional[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    cols = [c.upper() for c in df.columns]
    if "CLOTURE" not in cols:
        return None
    if symbol and "CODE" in cols:
        match = df[df["CODE"].astype(str).str.upper() == symbol.upper()]
        if not match.empty:
            return float(match["CLOTURE"].iloc[-1])
    return float(df["CLOTURE"].iloc[-1])


def load_forecast_module():
    path = os.path.join(ROOT, "cahier-de-charges-code_lab2.0", "src", "infer_five_day_forecast.py")
    return load_module(path, "infer_five_day_forecast")


def load_historical_data(forecast_module) -> pd.DataFrame:
    data_paths = (
        glob.glob(os.path.join(ROOT, "cahier-de-charges-code_lab2.0", "histo_cotation_*.csv"))
        + glob.glob(os.path.join(ROOT, "cahier-de-charges-code_lab2.0", "histo_cotation_*.txt"))
    )
    raw = forecast_module.load_bvmt_files(sorted(data_paths))
    normalized = forecast_module.normalize_market_data(raw)
    return normalized


def resolve_model_dir(company: Optional[str], symbol: Optional[str]) -> Optional[str]:
    base = os.path.join(ROOT, "cahier-de-charges-code_lab2.0", "models")
    if not os.path.isdir(base):
        return None

    candidates = []
    if symbol:
        candidates.append(symbol)
    if company:
        candidates.append(company)

    for name in candidates:
        path = os.path.join(base, str(name).strip())
        if os.path.isdir(path):
            return path

    # Case-insensitive fallback
    wanted = (symbol or company or "").upper()
    if wanted:
        for entry in os.listdir(base):
            if entry.upper() == wanted:
                return os.path.join(base, entry)

    return None


def find_any_model_dir() -> Optional[str]:
    base = os.path.join(ROOT, "cahier-de-charges-code_lab2.0", "models")
    if not os.path.isdir(base):
        return None
    for entry in os.listdir(base):
        path = os.path.join(base, entry)
        if os.path.isdir(path):
            return path
    return None


def compute_recent_prediction_errors(
    forecast_module,
    models: Dict[str, Any],
    company_df: pd.DataFrame,
    config,
    lookback: int = 3,
) -> List[float]:
    errors: List[float] = []
    if company_df.shape[0] < lookback + 5:
        return errors

    for idx in range(lookback, 0, -1):
        history = company_df.iloc[:-idx].copy()
        if history.empty:
            continue
        try:
            forecast = forecast_module.recursive_forecast(models, history, 1, config)
            pred = float(forecast["pred_close"].iloc[0])
            actual = float(company_df["CLOSE"].iloc[-idx])
            errors.append(abs(actual - pred))
        except Exception as exc:
            logger.warning("Backtest prediction error failed: %s", exc)
            continue
    return errors


def run_sentiment_analysis(company: Optional[str]) -> Optional[Dict[str, Any]]:
    if not company:
        return None
    path = os.path.join(ROOT, "Data Scrapping", "sentimental_analysis.py")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "IHEC-CODELAB", "Data Scrapping", "sentimental_analysis.py")
    if not os.path.exists(path):
        logger.warning("Sentiment module not found.")
        return None

    try:
        module = load_module(path, "sentimental_analysis")
    except Exception as exc:
        logger.warning("Sentiment module failed to load: %s", exc)
        return None
    analyzer_cls = getattr(module, "FinancialSentimentAnalyzer", None)
    if analyzer_cls is None:
        logger.warning("FinancialSentimentAnalyzer not found.")
        return None

    analyzer = analyzer_cls()
    return safe_call(analyzer.analyze_enterprise_sentiment, company, default=None, name="sentiment")


def load_models_safe(model_dir: str) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    xgb_path = os.path.join(model_dir, "xgboost.json")
    if os.path.exists(xgb_path):
        if importlib.util.find_spec("xgboost") is None:
            logger.warning("xgboost not installed; skipping XGBoost model.")
        else:
            from xgboost import XGBRegressor

            xgb = XGBRegressor()
            xgb.load_model(xgb_path)
            models["XGBoost"] = xgb

    gb_path = os.path.join(model_dir, "gradient_boosting.pkl")
    if os.path.exists(gb_path):
        if importlib.util.find_spec("joblib") is None:
            logger.warning("joblib not installed; skipping GradientBoosting model.")
        else:
            import joblib

            models["GradientBoosting"] = joblib.load(gb_path)

    return models


def run_anomaly_detection(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    path = os.path.join(ROOT, "anomalie", "anomaly_detector.py")
    if not os.path.exists(path):
        logger.warning("Anomaly detector not found.")
        return None
    module = load_module(path, "anomaly_detector")
    run_fn = getattr(module, "run_anomaly_detection", None)
    result = safe_call(run_fn, df, default=None, name="anomaly_detection")
    if isinstance(result, tuple):
        return result[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision engine demo")
    parser.add_argument("--symbol", default=None, help="Ticker code, e.g., BIAT")
    parser.add_argument(
        "--company",
        default="ATTIJARI BANK",
        help="Company name used for demo (default: ATTIJARI BANK)",
    )
    parser.add_argument(
        "--alert-stock",
        default="BIAT",
        help="Secondary stock for anomaly alerts (default: BIAT)",
    )
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--holdings", type=float, default=100.0)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    scraped = None
    company = args.company
    symbol = args.symbol or args.company
    current_price = None

    forecast_module = load_forecast_module()
    hist = safe_call(load_historical_data, forecast_module, default=None, name="load_historical_data")
    forecast_df = None
    prediction_errors: List[float] = []

    if hist is not None:
        company_df = hist[hist["COMPANY"].str.upper() == str(company).upper()].copy()
        if company_df.empty and symbol:
            company_df = hist[hist["CODE"].str.upper() == symbol.upper()].copy()

        if not company_df.empty:
            model_dir = resolve_model_dir(company, symbol)
            if model_dir:
                try:
                    models = load_models_safe(model_dir)
                    if not models:
                        raise RuntimeError("No usable models found in model directory.")
                    config = forecast_module.FeatureConfig(n_lags=5, volatility_window=10)
                    company_df = forecast_module.drop_non_trading_days(company_df)
                    forecast_df = forecast_module.recursive_forecast(models, company_df, args.horizon, config)
                    prediction_errors = compute_recent_prediction_errors(
                        forecast_module, models, company_df, config, lookback=3
                    )
                    if current_price is None and not company_df.empty:
                        current_price = float(company_df["CLOSE"].iloc[-1])
                except Exception as exc:
                    logger.warning("Forecast failed: %s", exc)
        if current_price is None and not company_df.empty:
            current_price = float(company_df["CLOSE"].iloc[-1])

    if current_price is None:
        current_price = 0.0

    sentiment = run_sentiment_analysis(company or symbol)

    anomalies_df = None
    alert_anomalies = None
    if hist is not None:
        history_subset = hist[hist["COMPANY"].str.upper() == str(company).upper()].copy()
        if history_subset.empty and symbol:
            history_subset = hist[hist["CODE"].str.upper() == symbol.upper()].copy()
        if not history_subset.empty:
            anomaly_source = pd.DataFrame({
                "CODE": history_subset["CODE"],
                "SEANCE": history_subset["DATE"],
                "CLOTURE": history_subset["CLOSE"],
                "QUANTITE_NEGOCIEE": history_subset["VOLUME"],
            }).tail(300)
            anomalies_df = run_anomaly_detection(anomaly_source)

        if args.alert_stock:
            alert_subset = hist[hist["COMPANY"].str.upper() == args.alert_stock.upper()].copy()
            if alert_subset.empty:
                alert_subset = hist[hist["CODE"].str.upper() == args.alert_stock.upper()].copy()
            if not alert_subset.empty:
                alert_source = pd.DataFrame({
                    "CODE": alert_subset["CODE"],
                    "SEANCE": alert_subset["DATE"],
                    "CLOTURE": alert_subset["CLOSE"],
                    "QUANTITE_NEGOCIEE": alert_subset["VOLUME"],
                }).tail(300)
                alert_anomalies = run_anomaly_detection(alert_source)

    engine = DecisionEngine(DecisionConfig(horizon=args.horizon))
    result = engine.run(
        symbol=symbol or "UNKNOWN",
        current_price=float(current_price),
        prediction_output=forecast_df,
        sentiment_output=sentiment,
        anomaly_output=anomalies_df,
        prediction_errors=prediction_errors,
        portfolio={"cash": args.cash, "holdings": args.holdings},
        sentiment_details=sentiment,
    )

    print("\n===== DECISION ENGINE RESULT =====")
    print(f"Symbol: {result['symbol']}")
    print(f"Company: {company}")
    print(f"Decision: {result['decision']}")
    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  - {k}: {v:.6f}")
    print("\nSignals:")
    for k, v in result["signals"].items():
        if k == "predicted_prices":
            print(f"  - {k}: {v}")
        else:
            print(f"  - {k}: {v}")
    print("\nPortfolio Projection (first 5 days):")
    for row in result["portfolio_projection"]:
        print(
            f"  Day {int(row['day'])}: price={row['price']:.3f}, total={row['total_value']:.2f}, "
            f"delta={row['delta']:.2f} ({row['pct_change']*100:.2f}%)"
        )

    if alert_anomalies is not None and not alert_anomalies.empty:
        alert_count = int(alert_anomalies["is_anomaly"].sum()) if "is_anomaly" in alert_anomalies.columns else 0
        print(f"\nAlert stock anomalies ({args.alert_stock}): {alert_count}")


if __name__ == "__main__":
    main()