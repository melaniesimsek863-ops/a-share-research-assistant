from __future__ import annotations

import pandas as pd

from a_share_ai.backtest.metrics import calculate_backtest_metrics
from a_share_ai.models import BacktestConfig


def run_simple_backtest(
    signals: pd.DataFrame,
    price_history: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    empty_equity = pd.DataFrame({"date": [], "equity": []})
    empty_trades = pd.DataFrame({"return": [], "holding_days": []})
    eligible_signals = signals[signals["can_buy"]] if not signals.empty else signals
    if eligible_signals.empty:
        return empty_equity, empty_trades, calculate_backtest_metrics(empty_equity, empty_trades)

    signal = eligible_signals.iloc[0]
    prices = price_history[
        (price_history["code"] == signal["code"])
        & (price_history["date"] >= signal["date"])
    ].sort_values("date").reset_index(drop=True)
    if prices.empty:
        return empty_equity, empty_trades, calculate_backtest_metrics(empty_equity, empty_trades)

    entry = prices.iloc[0]
    exit_row = prices.iloc[-1]
    gross_return = exit_row["close"] / entry["close"] - 1
    cost_rate = config.commission_rate * 2 + config.slippage_rate * 2
    net_return = gross_return - cost_rate
    position = float(signal["suggested_position"])

    equity = prices[["date"]].copy()
    equity["asset_return"] = prices["close"] / entry["close"] - 1
    equity["equity"] = config.initial_cash * (1 + (equity["asset_return"] - cost_rate) * position)
    baseline = pd.DataFrame({"date": [entry["date"]], "asset_return": [0.0], "equity": [config.initial_cash]})
    equity = pd.concat([baseline, equity], ignore_index=True)

    trades = pd.DataFrame(
        [
            {
                "code": signal["code"],
                "entry_date": entry["date"],
                "exit_date": exit_row["date"],
                "return": net_return,
                "holding_days": len(prices),
            }
        ]
    )
    metrics = calculate_backtest_metrics(equity, trades)
    return equity, trades, metrics
