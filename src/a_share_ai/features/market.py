from __future__ import annotations

import pandas as pd

from a_share_ai.features.technical import add_technical_features


def calculate_market_regime(index_bars: pd.DataFrame) -> dict[str, float | str]:
    enriched = add_technical_features(index_bars)
    latest = enriched.iloc[-1]
    score = 0.0
    if bool(latest["above_ma20"]):
        score += 0.4
    if bool(latest["ma_bullish"]):
        score += 0.4
    if latest["return_20d"] > 0:
        score += 0.2

    if score >= 0.7:
        label = "uptrend"
    elif score <= 0.2:
        label = "weak"
    else:
        label = "neutral"
    return {"label": label, "score": round(score, 4)}
