from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def demo_stock_info() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "平安银行",
                "board": "main",
                "listing_date": date(1991, 4, 3),
                "is_st": False,
                "is_delisting_risk": False,
                "is_suspended": False,
                "avg_turnover_20d": 500_000_000,
            },
            {
                "code": "300001",
                "name": "特锐德",
                "board": "chinext",
                "listing_date": date(2009, 10, 30),
                "is_st": False,
                "is_delisting_risk": False,
                "is_suspended": False,
                "avg_turnover_20d": 180_000_000,
            },
            {
                "code": "600001",
                "name": "ST测试",
                "board": "main",
                "listing_date": date(2000, 1, 1),
                "is_st": True,
                "is_delisting_risk": False,
                "is_suspended": False,
                "avg_turnover_20d": 200_000_000,
            },
            {
                "code": "688001",
                "name": "科创测试",
                "board": "star",
                "listing_date": date(2019, 7, 22),
                "is_st": False,
                "is_delisting_risk": False,
                "is_suspended": False,
                "avg_turnover_20d": 300_000_000,
            },
            {
                "code": "000002",
                "name": "新股测试",
                "board": "main",
                "listing_date": date(2026, 6, 15),
                "is_st": False,
                "is_delisting_risk": False,
                "is_suspended": False,
                "avg_turnover_20d": 300_000_000,
            },
            {
                "code": "300002",
                "name": "停牌测试",
                "board": "chinext",
                "listing_date": date(2010, 1, 1),
                "is_st": False,
                "is_delisting_risk": False,
                "is_suspended": True,
                "avg_turnover_20d": 300_000_000,
            },
            {
                "code": "000003",
                "name": "低流动性",
                "board": "main",
                "listing_date": date(1991, 1, 1),
                "is_st": False,
                "is_delisting_risk": False,
                "is_suspended": False,
                "avg_turnover_20d": 20_000_000,
            },
        ]
    )


def demo_price_history(code: str = "000001", days: int = 140, trend: float = 0.002) -> pd.DataFrame:
    dates = pd.date_range(end="2026-07-29", periods=days, freq="B")
    base = 10.0
    closes = base * np.cumprod(np.repeat(1 + trend, days))
    opens = closes * 0.995
    highs = closes * 1.02
    lows = closes * 0.98
    volumes = np.linspace(10_000_000, 20_000_000, days)
    amounts = volumes * closes
    return pd.DataFrame(
        {
            "date": dates.date,
            "code": code,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "amount": amounts,
        }
    )
