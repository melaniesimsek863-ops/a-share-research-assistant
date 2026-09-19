from __future__ import annotations

import pandas as pd


def score_medium_term(latest_rows: pd.DataFrame) -> pd.DataFrame:
    data = latest_rows.copy()
    score = pd.Series(0.0, index=data.index)
    score += data["above_ma20"].astype(float) * 20
    score += data["ma_bullish"].astype(float) * 25
    score += (data["return_20d"].clip(-0.1, 0.2) + 0.1) / 0.3 * 20
    score += (data["return_60d"].clip(-0.2, 0.4) + 0.2) / 0.6 * 20
    score += (data["drawdown_60d"].clip(-0.25, 0) + 0.25) / 0.25 * 15
    data["medium_term_score"] = score.clip(0, 100).round(2)
    return data
