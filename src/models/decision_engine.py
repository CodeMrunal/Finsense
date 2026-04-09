"""
Financial decision engine for BUY / SELL / HOLD recommendations.
"""
from typing import Dict, Union


Number = Union[int, float]


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _normalize_prediction(prediction: Number) -> float:
    """
    Normalize prediction signal to 0-100.
    Assumes input is expected return in percentage points or decimal return.
    """
    p = float(prediction)
    if -1.0 <= p <= 1.0:
        p = p * 100.0
    # Map [-20, +20] -> [0, 100]
    return _clip((p + 20.0) * 2.5)


def _normalize_sentiment(sentiment: Union[str, Number]) -> float:
    """Convert sentiment to 0-100 score."""
    if isinstance(sentiment, (int, float)):
        s = float(sentiment)
        if 0.0 <= s <= 1.0:
            return _clip(s * 100.0)
        return _clip(s)

    text = str(sentiment).strip().lower()
    mapping = {
        "very_bearish": 10.0,
        "bearish": 25.0,
        "negative": 30.0,
        "neutral": 50.0,
        "positive": 70.0,
        "bullish": 75.0,
        "very_bullish": 90.0,
    }
    return mapping.get(text, 50.0)


def _stability_score(volatility: Number, trend: Union[str, Number]) -> float:
    """
    Compute 0-100 stability score from volatility and trend.
    Lower volatility and stable/upward trend increase stability.
    """
    vol = float(volatility)
    if vol <= 2.0:
        vol = vol * 100.0

    # Volatility contribution: lower vol => higher score
    vol_score = _clip(100.0 - vol * 2.0)

    # Trend contribution
    if isinstance(trend, str):
        t = trend.lower()
        if "up" in t or "bull" in t:
            trend_score = 70.0
        elif "stable" in t or "side" in t or "neutral" in t:
            trend_score = 60.0
        else:
            trend_score = 40.0
    else:
        t = float(trend)
        trend_score = _clip(50.0 + (t * 200.0))

    return _clip(0.6 * vol_score + 0.4 * trend_score)


def compute_decision(
    prediction: Number,
    rsi: Number,
    sentiment: Union[str, Number],
    volatility: Number,
    trend: Union[str, Number]
) -> Dict[str, Union[str, float]]:
    """
    Compute BUY/SELL/HOLD with confidence and risk.

    score =
      0.4 * prediction +
      0.2 * (100 - rsi) +
      0.2 * sentiment_score +
      0.2 * stability_score
    """
    pred_score = _normalize_prediction(prediction)
    rsi_score = _clip(100.0 - float(rsi))
    sentiment_score = _normalize_sentiment(sentiment)
    stability = _stability_score(volatility, trend)

    score = (
        0.4 * pred_score +
        0.2 * rsi_score +
        0.2 * sentiment_score +
        0.2 * stability
    )

    if score > 70:
        decision = "BUY"
    elif score >= 40:
        decision = "HOLD"
    else:
        decision = "SELL"

    # Confidence: baseline + signal strength factors
    base = 50.0
    pred_strength = abs(pred_score - 50.0) / 50.0 * 20.0
    sentiment_strength = abs(sentiment_score - 50.0) / 50.0 * 15.0
    if isinstance(trend, str):
        t_up = "up" in trend.lower()
        t_down = "down" in trend.lower()
    else:
        t_up = float(trend) > 0
        t_down = float(trend) < 0
    trend_align = 10.0 if ((decision == "BUY" and t_up) or (decision == "SELL" and t_down)) else 0.0
    confidence = _clip(base + pred_strength + sentiment_strength + trend_align, 0.0, 100.0)

    vol = float(volatility)
    if vol <= 2.0:
        vol = vol * 100.0
    if vol >= 30:
        risk = "High"
    elif vol >= 15:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "decision": decision,
        "confidence": round(confidence, 2),
        "risk_level": risk,
        "score": round(score, 2),
        "components": {
            "prediction_score": round(pred_score, 2),
            "rsi_score": round(rsi_score, 2),
            "sentiment_score": round(sentiment_score, 2),
            "stability_score": round(stability, 2),
        },
    }

