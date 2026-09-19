from __future__ import annotations

import pandas as pd


def _labels(row: pd.Series) -> list[str]:
    labels: list[str] = []
    if row["volume_ratio_5_20"] >= 1.5 and row["close"] > row["ma20"]:
        labels.append("放量且收盘在20日均线上方")
    if row["close"] >= row["ma20"] and row["drawdown_60d"] > -0.08:
        labels.append("收盘在20日均线上方且60日回撤大于-8%")
    if row["return_20d"] < 0.18:
        labels.append("20日涨幅低于18%")
    return labels


def score_timing(latest_rows: pd.DataFrame) -> pd.DataFrame:
    data = latest_rows.copy()
    labels = data.apply(_labels, axis=1)
    data["signal_labels"] = labels
    data["timing_score"] = labels.map(lambda values: min(len(values) * 30, 100)).astype(float)
    return data
