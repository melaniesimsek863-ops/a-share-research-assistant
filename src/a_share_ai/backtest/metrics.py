from __future__ import annotations

import math

import pandas as pd


def calculate_backtest_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "trade_count": 0,
            "avg_holding_days": 0.0,
        }

    equity = equity_curve["equity"].astype(float)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    trade_count = len(trades)
    if trade_count == 0:
        win_rate = payoff_ratio = avg_holding_days = 0.0
    else:
        wins = trades[trades["return"] > 0]["return"]
        losses = trades[trades["return"] <= 0]["return"]
        win_rate = len(wins) / trade_count
        if wins.empty or losses.empty:
            payoff_ratio = 0.0
        else:
            average_loss = abs(float(losses.mean()))
            raw_payoff_ratio = float(wins.mean()) / average_loss if average_loss else 0.0
            payoff_ratio = raw_payoff_ratio if math.isfinite(raw_payoff_ratio) else 0.0
        avg_holding_days = float(trades["holding_days"].mean())

    return {
        "total_return": round(float(total_return), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "win_rate": round(float(win_rate), 4),
        "payoff_ratio": round(float(payoff_ratio), 4),
        "trade_count": int(trade_count),
        "avg_holding_days": round(float(avg_holding_days), 2),
    }
