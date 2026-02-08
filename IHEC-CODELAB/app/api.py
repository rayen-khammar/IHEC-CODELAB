from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.database import engine, init_db

import glob
import importlib.util
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(ROOT)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_forecast_module():
    path = os.path.join(WORKSPACE_ROOT, "cahier-de-charges-code_lab2.0", "src", "infer_five_day_forecast.py")
    return load_module(path, "infer_five_day_forecast")


def load_historical_data(forecast_module) -> pd.DataFrame:
    data_paths = (
        glob.glob(os.path.join(WORKSPACE_ROOT, "cahier-de-charges-code_lab2.0", "histo_cotation_*.csv"))
        + glob.glob(os.path.join(WORKSPACE_ROOT, "cahier-de-charges-code_lab2.0", "histo_cotation_*.txt"))
    )
    raw = forecast_module.load_bvmt_files(sorted(data_paths))
    normalized = forecast_module.normalize_market_data(raw)
    return normalized


def resolve_model_dir(company: Optional[str], symbol: Optional[str]) -> Optional[str]:
    base = os.path.join(WORKSPACE_ROOT, "cahier-de-charges-code_lab2.0", "models")
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

    wanted = (symbol or company or "").upper()
    if wanted:
        for entry in os.listdir(base):
            if entry.upper() == wanted:
                return os.path.join(base, entry)

    return None


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
    path = os.path.join(WORKSPACE_ROOT, "Data Scrapping", "sentimental_analysis.py")
    if not os.path.exists(path):
        path = os.path.join(WORKSPACE_ROOT, "IHEC-CODELAB", "Data Scrapping", "sentimental_analysis.py")
    if not os.path.exists(path):
        return None

    try:
        module = load_module(path, "sentimental_analysis")
    except Exception as exc:
        logger.warning("Sentiment module failed to load: %s", exc)
        return None

    analyzer_cls = getattr(module, "FinancialSentimentAnalyzer", None)
    if analyzer_cls is None:
        return None

    analyzer = analyzer_cls()
    try:
        return analyzer.analyze_enterprise_sentiment(company)
    except Exception as exc:
        logger.warning("Sentiment analysis failed: %s", exc)
        return None


def run_anomaly_detection(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    path = os.path.join(WORKSPACE_ROOT, "anomalie", "anomaly_detector.py")
    if not os.path.exists(path):
        return None
    module = load_module(path, "anomaly_detector")
    run_fn = getattr(module, "run_anomaly_detection", None)
    try:
        result = run_fn(df)
    except Exception as exc:
        logger.warning("Anomaly detection failed: %s", exc)
        return None
    if isinstance(result, tuple):
        return result[0]
    return result


def build_decision_demo(company: str, alert_stock: Optional[str], cash: float, holdings: float, horizon: int):
    if WORKSPACE_ROOT not in sys.path:
        sys.path.insert(0, WORKSPACE_ROOT)

    from decision_engine import DecisionEngine, DecisionConfig

    forecast_module = load_forecast_module()
    hist = load_historical_data(forecast_module)

    company_df = hist[hist["COMPANY"].str.upper() == company.upper()].copy()
    if company_df.empty:
        company_df = hist[hist["CODE"].str.upper() == company.upper()].copy()
    if company_df.empty:
        raise ValueError("Company not found in historical data")

    model_dir = resolve_model_dir(company, company)
    if not model_dir:
        raise ValueError("Model directory not found for company")

    models = load_models_safe(model_dir)
    if not models:
        raise ValueError("No usable models found")

    config = forecast_module.FeatureConfig(n_lags=5, volatility_window=10)
    company_df = forecast_module.drop_non_trading_days(company_df)
    forecast_df = forecast_module.recursive_forecast(models, company_df, horizon, config)
    prediction_errors = compute_recent_prediction_errors(forecast_module, models, company_df, config, lookback=3)

    current_price = float(company_df["CLOSE"].iloc[-1])

    sentiment = run_sentiment_analysis(company)

    anomaly_source = pd.DataFrame({
        "CODE": company_df["CODE"],
        "SEANCE": company_df["DATE"],
        "CLOTURE": company_df["CLOSE"],
        "QUANTITE_NEGOCIEE": company_df["VOLUME"],
    }).tail(300)
    anomalies_df = run_anomaly_detection(anomaly_source)

    alert_anomalies = None
    if alert_stock:
        alert_df = hist[hist["COMPANY"].str.upper() == alert_stock.upper()].copy()
        if alert_df.empty:
            alert_df = hist[hist["CODE"].str.upper() == alert_stock.upper()].copy()
        if not alert_df.empty:
            alert_source = pd.DataFrame({
                "CODE": alert_df["CODE"],
                "SEANCE": alert_df["DATE"],
                "CLOTURE": alert_df["CLOSE"],
                "QUANTITE_NEGOCIEE": alert_df["VOLUME"],
            }).tail(300)
            alert_anomalies = run_anomaly_detection(alert_source)

    engine = DecisionEngine(DecisionConfig(horizon=horizon))
    result = engine.run(
        symbol=company,
        current_price=current_price,
        prediction_output=forecast_df,
        sentiment_output=sentiment,
        anomaly_output=anomalies_df,
        prediction_errors=prediction_errors,
        portfolio={"cash": cash, "holdings": holdings},
        sentiment_details=sentiment,
    )

    alert_count = 0
    if alert_anomalies is not None and "is_anomaly" in alert_anomalies.columns:
        alert_count = int(alert_anomalies["is_anomaly"].sum())

    result["alert_stock"] = alert_stock
    result["alert_anomaly_count"] = alert_count
    result["company"] = company
    return result

app = FastAPI(title="BVMT Live API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def root():
    return {"status": "ok", "message": "BVMT Live API running"}

@app.get("/codes")
def codes():
    q = text("""
        SELECT DISTINCT code
        FROM ticks
        WHERE code IS NOT NULL
        ORDER BY code ASC
        LIMIT 500
    """)
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()

    return [r[0] for r in rows]

@app.get("/latest")
def latest(code: str):
    q = text("""
        SELECT *
        FROM ticks
        WHERE code = :code
        ORDER BY id DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(q, {"code": code}).fetchone()

    return dict(row._mapping) if row else {}

@app.get("/history")
def history(code: str, limit: int = 200):
    q = text("""
        SELECT *
        FROM ticks
        WHERE code = :code
        ORDER BY id DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"code": code, "limit": limit}).fetchall()

    return [dict(r._mapping) for r in rows]


@app.get("/decision_demo")
def decision_demo(
    company: str = "ATTIJARI BANK",
    alert_stock: Optional[str] = "BIAT",
    cash: float = 10000.0,
    holdings: float = 100.0,
    horizon: int = 5,
):
    """Run the decision engine demo using cahier-de-charges data."""
    return build_decision_demo(company, alert_stock, cash, holdings, horizon)
