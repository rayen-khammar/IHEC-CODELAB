"""Decision & Portfolio Engine for BVMT Intelligent Trading Assistant.

This module integrates multiple unreliable signals (prediction, variance,
sentiment, anomaly, risk preference) into a robust decision pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def safe_call(
    func: Optional[Callable[..., Any]],
    *args: Any,
    default: Any = None,
    name: str = "",
    **kwargs: Any,
) -> Any:
    if func is None:
        return default
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        label = name or getattr(func, "__name__", "<callable>")
        logger.warning("Module call failed (%s): %s", label, exc)
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(" ", "").replace(",", "."))
        except ValueError:
            return default
    return default


def _coerce_float_list(values: Iterable[Any]) -> List[float]:
    result: List[float] = []
    for val in values:
        fval = _to_float(val)
        if fval is not None:
            result.append(float(fval))
    return result


def _extract_dataframe_column(data: Any, candidates: Iterable[str]) -> Optional[List[float]]:
    if data is None or not hasattr(data, "columns"):
        return None
    columns = [str(c) for c in getattr(data, "columns", [])]
    for col in candidates:
        if col in columns:
            try:
                series = data[col]
                return _coerce_float_list(getattr(series, "tolist", lambda: list(series))())
            except Exception:
                return None
    return None


def normalize_predictions(prediction_output: Any, horizon: int) -> List[float]:
    if prediction_output is None:
        return []

    if isinstance(prediction_output, dict):
        for key in ("predicted_prices", "predictions", "forecast", "preds"):
            if key in prediction_output:
                return _coerce_float_list(prediction_output.get(key, []))[:horizon]
        for key in ("forecast_df", "forecast_dataframe"):
            if key in prediction_output:
                values = _extract_dataframe_column(prediction_output[key], ["pred_close", "prediction", "yhat"])
                if values:
                    return values[:horizon]

    if hasattr(prediction_output, "columns"):
        values = _extract_dataframe_column(prediction_output, ["pred_close", "prediction", "yhat"])
        if values:
            return values[:horizon]

    if isinstance(prediction_output, (list, tuple)):
        return _coerce_float_list(prediction_output)[:horizon]

    return []


def estimate_variance_from_predictions(current_price: float, predicted_prices: List[float]) -> Optional[float]:
    if not predicted_prices or current_price <= 0:
        return None
    returns = [(p - current_price) / current_price for p in predicted_prices if p is not None]
    if len(returns) < 2:
        return None
    mu = mean(returns)
    variance = mean([(r - mu) ** 2 for r in returns])
    return float(variance)


def normalize_variance(variance_output: Any, current_price: float, predicted_prices: List[float], default: float) -> float:
    if isinstance(variance_output, dict):
        for key in ("variance", "pred_variance", "prediction_variance", "uncertainty"):
            if key in variance_output:
                val = _to_float(variance_output.get(key))
                if val is not None:
                    return max(0.0, float(val))

    val = _to_float(variance_output)
    if val is not None:
        return max(0.0, float(val))

    estimated = estimate_variance_from_predictions(current_price, predicted_prices)
    if estimated is not None:
        return max(0.0, float(estimated))

    return max(0.0, float(default))


def normalize_sentiment(sentiment_output: Any, default: float = 0.0) -> float:
    if isinstance(sentiment_output, dict):
        for key in ("overall_sentiment_score", "sentiment_score", "score"):
            if key in sentiment_output:
                val = _to_float(sentiment_output.get(key))
                if val is not None:
                    return float(max(-1.0, min(1.0, val)))
        if "articles" in sentiment_output and isinstance(sentiment_output["articles"], list):
            scores = _coerce_float_list([a.get("score") for a in sentiment_output["articles"] if isinstance(a, dict)])
            if scores:
                avg = sum(scores) / len(scores)
                return float(max(-1.0, min(1.0, avg)))

    val = _to_float(sentiment_output)
    if val is not None:
        return float(max(-1.0, min(1.0, val)))

    return float(default)


def _bool_to_score(flag: Any) -> Optional[float]:
    if isinstance(flag, bool):
        return 1.0 if flag else 0.0
    return None


def normalize_anomaly(anomaly_output: Any, default: float = 0.0) -> float:
    if isinstance(anomaly_output, dict):
        for key in ("anomaly_score", "score"):
            if key in anomaly_output:
                val = _to_float(anomaly_output.get(key))
                if val is not None:
                    return float(max(0.0, min(1.0, val)))
        for key in ("is_anomaly", "anomaly_flag"):
            if key in anomaly_output:
                val = _bool_to_score(anomaly_output.get(key))
                if val is not None:
                    return float(val)
        if "anomalies" in anomaly_output and isinstance(anomaly_output["anomalies"], list):
            return float(min(1.0, max(0.0, len(anomaly_output["anomalies"]) / 3.0)))

    if hasattr(anomaly_output, "columns"):
        if "is_anomaly" in anomaly_output.columns:
            try:
                values = anomaly_output["is_anomaly"].tolist()
                if values:
                    return float(1.0 if bool(values[-1]) else 0.0)
            except Exception:
                pass

    val = _to_float(anomaly_output)
    if val is not None:
        return float(max(0.0, min(1.0, val)))

    return float(default)


def estimate_model_anomaly(error_history: Iterable[float]) -> float:
    errors = _coerce_float_list(error_history)
    if len(errors) < 3:
        return 0.0
    recent = errors[-1]
    avg = mean(errors)
    sigma = pstdev(errors) or 0.0
    if sigma == 0:
        return 1.0 if recent > avg else 0.0
    z = (recent - avg) / sigma
    if z >= 2.0:
        return 1.0
    if z <= 0:
        return 0.0
    return float(min(1.0, max(0.0, z / 2.0)))


def combine_anomaly_scores(classic_score: float, model_score: float) -> float:
    return float(max(0.0, min(1.0, max(classic_score, model_score))))


@dataclass
class SignalBundle:
    current_price: float
    predicted_prices: List[float]
    variance: float
    sentiment_score: float
    anomaly_score: float
    user_risk_lambda: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionConfig:
    horizon: int = 5
    sentiment_alpha: float = 0.5
    anomaly_weight: float = 2.0
    buy_threshold: float = 0.01
    sell_threshold: float = -0.01
    default_lambda: float = 0.5
    default_variance: float = 0.0004
    default_anomaly: float = 0.0


class DecisionEngine:
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()

    def build_signals(
        self,
        current_price: float,
        prediction_output: Any = None,
        variance_output: Any = None,
        sentiment_output: Any = None,
        anomaly_output: Any = None,
        user_risk_output: Any = None,
        prediction_errors: Optional[Iterable[float]] = None,
    ) -> SignalBundle:
        predicted_prices = normalize_predictions(prediction_output, self.config.horizon)

        variance = normalize_variance(
            variance_output,
            current_price=current_price,
            predicted_prices=predicted_prices,
            default=self.config.default_variance,
        )
        sentiment_score = normalize_sentiment(sentiment_output, default=0.0)
        classic_anomaly = normalize_anomaly(anomaly_output, default=self.config.default_anomaly)
        model_anomaly = estimate_model_anomaly(prediction_errors or [])
        anomaly_score = combine_anomaly_scores(classic_anomaly, model_anomaly)

        user_lambda = _to_float(user_risk_output, default=self.config.default_lambda)
        if user_lambda is None:
            user_lambda = self.config.default_lambda
        user_lambda = max(0.0, float(user_lambda))

        return SignalBundle(
            current_price=float(current_price),
            predicted_prices=predicted_prices,
            variance=variance,
            sentiment_score=sentiment_score,
            anomaly_score=anomaly_score,
            user_risk_lambda=user_lambda,
            metadata={
                "classic_anomaly_score": classic_anomaly,
                "model_anomaly_score": model_anomaly,
            },
        )

    def compute_decision(self, signals: SignalBundle) -> Tuple[str, Dict[str, float]]:
        if signals.predicted_prices:
            target_price = signals.predicted_prices[0]
        else:
            target_price = signals.current_price

        if signals.current_price <= 0:
            expected_return = 0.0
        else:
            expected_return = (target_price - signals.current_price) / signals.current_price

        adjusted_return = expected_return * (1.0 + self.config.sentiment_alpha * signals.sentiment_score)

        anomaly_penalty = 1.0 + self.config.anomaly_weight * signals.anomaly_score
        risk = signals.variance * anomaly_penalty

        utility_score = adjusted_return - signals.user_risk_lambda * risk

        if utility_score > self.config.buy_threshold:
            decision = "BUY"
        elif utility_score < self.config.sell_threshold:
            decision = "SELL"
        else:
            decision = "HOLD"

        if signals.anomaly_score >= 0.7 and decision == "BUY":
            decision = "HOLD"
        elif signals.anomaly_score >= 0.85 and decision == "HOLD":
            decision = "SELL"

        metrics = {
            "expected_return": float(expected_return),
            "adjusted_return": float(adjusted_return),
            "risk": float(risk),
            "utility_score": float(utility_score),
            "anomaly_penalty": float(anomaly_penalty),
        }
        return decision, metrics

    def simulate_portfolio(
        self,
        portfolio: Optional[Dict[str, float]],
        current_price: float,
        predicted_prices: List[float],
    ) -> List[Dict[str, float]]:
        portfolio = portfolio or {}
        cash = _to_float(portfolio.get("cash"), 0.0) or 0.0
        holdings = _to_float(portfolio.get("holdings"), 0.0) or 0.0

        base_value = cash + holdings * current_price
        last_price = current_price
        projections: List[Dict[str, float]] = []

        for step in range(1, self.config.horizon + 1):
            if step <= len(predicted_prices) and predicted_prices[step - 1] is not None:
                price = predicted_prices[step - 1]
            else:
                price = last_price

            holdings_value = holdings * price
            total_value = cash + holdings_value
            delta = total_value - base_value
            pct_change = (total_value / base_value - 1.0) if base_value > 0 else 0.0

            projections.append(
                {
                    "day": float(step),
                    "price": float(price),
                    "cash": float(cash),
                    "holdings": float(holdings),
                    "holdings_value": float(holdings_value),
                    "total_value": float(total_value),
                    "delta": float(delta),
                    "pct_change": float(pct_change),
                }
            )

            last_price = price

        return projections

    def build_explanation(
        self,
        symbol: str,
        signals: SignalBundle,
        decision: str,
        metrics: Dict[str, float],
        sentiment_details: Optional[Dict[str, Any]] = None,
        anomaly_details: Optional[Dict[str, Any]] = None,
        similar_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        explanation: Dict[str, Any] = {
            "symbol": symbol,
            "decision": decision,
            "expected_return": metrics.get("expected_return", 0.0),
            "adjusted_return": metrics.get("adjusted_return", 0.0),
            "risk": metrics.get("risk", 0.0),
            "utility_score": metrics.get("utility_score", 0.0),
            "sentiment_score": signals.sentiment_score,
            "anomaly_score": signals.anomaly_score,
            "variance": signals.variance,
            "notes": [],
        }

        explanation["notes"].append(
            "Expected return is derived from the next-day prediction versus current price."
        )
        explanation["notes"].append(
            "Risk combines prediction variance and anomaly penalty."
        )

        if signals.sentiment_score != 0:
            explanation["notes"].append("Sentiment adjusted the expected return.")

        if signals.anomaly_score >= 0.7:
            explanation["notes"].append("High anomaly score downgraded the recommendation.")

        if sentiment_details:
            explanation["sentiment_details"] = sentiment_details
        if anomaly_details:
            explanation["anomaly_details"] = anomaly_details

        if similar_cases:
            explanation["similar_cases"] = similar_cases
        else:
            explanation["similar_cases"] = []

        return explanation

    def run(
        self,
        symbol: str,
        current_price: float,
        prediction_output: Any = None,
        variance_output: Any = None,
        sentiment_output: Any = None,
        anomaly_output: Any = None,
        user_risk_output: Any = None,
        portfolio: Optional[Dict[str, float]] = None,
        prediction_errors: Optional[Iterable[float]] = None,
        sentiment_details: Optional[Dict[str, Any]] = None,
        anomaly_details: Optional[Dict[str, Any]] = None,
        similar_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        signals = self.build_signals(
            current_price=current_price,
            prediction_output=prediction_output,
            variance_output=variance_output,
            sentiment_output=sentiment_output,
            anomaly_output=anomaly_output,
            user_risk_output=user_risk_output,
            prediction_errors=prediction_errors,
        )

        decision, metrics = self.compute_decision(signals)
        portfolio_projection = self.simulate_portfolio(
            portfolio=portfolio,
            current_price=signals.current_price,
            predicted_prices=signals.predicted_prices,
        )
        explanation = self.build_explanation(
            symbol=symbol,
            signals=signals,
            decision=decision,
            metrics=metrics,
            sentiment_details=sentiment_details,
            anomaly_details=anomaly_details,
            similar_cases=similar_cases,
        )

        return {
            "symbol": symbol,
            "signals": asdict(signals),
            "decision": decision,
            "metrics": metrics,
            "portfolio_projection": portfolio_projection,
            "explanation": explanation,
        }


def run_decision_pipeline(
    symbol: str,
    current_price: float,
    prediction_provider: Optional[Callable[..., Any]] = None,
    sentiment_provider: Optional[Callable[..., Any]] = None,
    anomaly_provider: Optional[Callable[..., Any]] = None,
    risk_provider: Optional[Callable[..., Any]] = None,
    portfolio: Optional[Dict[str, float]] = None,
    prediction_errors: Optional[Iterable[float]] = None,
    engine: Optional[DecisionEngine] = None,
    **provider_kwargs: Any,
) -> Dict[str, Any]:
    """High-level pipeline with defensive module calls."""

    engine = engine or DecisionEngine()

    prediction_output = safe_call(prediction_provider, symbol, **provider_kwargs, default=None, name="prediction")
    sentiment_output = safe_call(sentiment_provider, symbol, **provider_kwargs, default=None, name="sentiment")
    anomaly_output = safe_call(anomaly_provider, symbol, **provider_kwargs, default=None, name="anomaly")
    user_risk_output = safe_call(risk_provider, symbol, **provider_kwargs, default=None, name="risk")

    return engine.run(
        symbol=symbol,
        current_price=current_price,
        prediction_output=prediction_output,
        sentiment_output=sentiment_output,
        anomaly_output=anomaly_output,
        user_risk_output=user_risk_output,
        portfolio=portfolio,
        prediction_errors=prediction_errors,
    )