from __future__ import annotations

import pandas as pd


def _to_python_bool(series: pd.Series) -> pd.Series:
    return series.map(bool).astype(object)


def add_technical_features(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    data = bars.sort_values(["code", "date"]).copy()
    grouped = data.groupby("code", group_keys=False)

    data["ma5"] = grouped["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    data["ma20"] = grouped["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    data["ma60"] = grouped["close"].transform(lambda x: x.rolling(60, min_periods=60).mean())
    data["return_20d"] = grouped["close"].transform(lambda x: x.pct_change(20))
    data["return_60d"] = grouped["close"].transform(lambda x: x.pct_change(60))
    data["volume_ma5"] = grouped["volume"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    data["volume_ma20"] = grouped["volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    data["volume_ratio_5_20"] = data["volume_ma5"] / data["volume_ma20"]
    rolling_high_60 = grouped["close"].transform(lambda x: x.rolling(60, min_periods=20).max())
    data["drawdown_60d"] = data["close"] / rolling_high_60 - 1
    data["above_ma20"] = _to_python_bool(data["close"] > data["ma20"])
    data["ma_bullish"] = _to_python_bool((data["ma5"] > data["ma20"]) & (data["ma20"] > data["ma60"]))
    return data
