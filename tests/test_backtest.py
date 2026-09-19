import math

import pandas as pd

from a_share_ai.backtest.engine import run_simple_backtest
from a_share_ai.backtest.metrics import calculate_backtest_metrics
from a_share_ai.config import load_config
from a_share_ai.reports.backtest_report import build_backtest_summary
from tests.fixtures.sample_data import sample_price_history


def test_calculate_backtest_metrics_from_equity_curve():
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "equity": [100.0, 110.0, 105.0, 120.0],
        }
    )
    trades = pd.DataFrame({"return": [0.1, -0.03, 0.05], "holding_days": [5, 3, 8]})
    metrics = calculate_backtest_metrics(equity, trades)

    assert metrics["total_return"] == 0.2
    assert round(metrics["max_drawdown"], 4) == -0.0455
    assert metrics["win_rate"] == 0.6667
    assert metrics["payoff_ratio"] == 2.5
    assert metrics["trade_count"] == 3
    assert metrics["avg_holding_days"] == 5.33


def test_run_simple_backtest_returns_summary():
    config = load_config("configs/default.yaml")
    prices = sample_price_history(days=40)
    signals = pd.DataFrame(
        [
            {
                "date": prices.iloc[0]["date"],
                "code": "000001",
                "suggested_position": 0.15,
                "can_buy": True,
            }
        ]
    )
    equity, trades, metrics = run_simple_backtest(signals, prices, config.backtest)

    assert not equity.empty
    assert not trades.empty
    assert metrics["trade_count"] == 1
    assert metrics["payoff_ratio"] == 0.0
    assert metrics["avg_holding_days"] == 40.0


def test_run_simple_backtest_enters_on_or_after_signal_date_and_applies_costs():
    config = load_config("configs/default.yaml")
    prices = sample_price_history(days=40)
    signal_date = prices.iloc[10]["date"]
    signals = pd.DataFrame(
        [
            {
                "date": signal_date,
                "code": "000001",
                "suggested_position": 0.15,
                "can_buy": True,
            }
        ]
    )

    equity, trades, _ = run_simple_backtest(signals, prices, config.backtest)

    eligible_prices = prices[prices["date"] >= signal_date]
    assert trades.iloc[0]["entry_date"] == eligible_prices.iloc[0]["date"]
    assert equity.iloc[0]["date"] == eligible_prices.iloc[0]["date"]
    gross_return = eligible_prices.iloc[-1]["close"] / eligible_prices.iloc[0]["close"] - 1
    cost_rate = config.backtest.commission_rate * 2 + config.backtest.slippage_rate * 2
    expected_equity = config.backtest.initial_cash * (1 + (gross_return - cost_rate) * 0.15)
    assert equity.iloc[-1]["equity"] == expected_equity


def test_run_simple_backtest_costs_reduce_flat_price_total_return():
    config = load_config("configs/default.yaml")
    prices = sample_price_history(days=5)
    prices["close"] = 10.0
    signals = pd.DataFrame(
        [
            {
                "date": prices.iloc[0]["date"],
                "code": "000001",
                "suggested_position": 0.15,
                "can_buy": True,
            }
        ]
    )

    equity, trades, metrics = run_simple_backtest(signals, prices, config.backtest)

    cost_rate = config.backtest.commission_rate * 2 + config.backtest.slippage_rate * 2
    expected_portfolio_return = -cost_rate * 0.15
    assert metrics["total_return"] == round(expected_portfolio_return, 4)
    assert metrics["max_drawdown"] == round(expected_portfolio_return, 4)
    assert trades.iloc[0]["return"] == -cost_rate
    assert equity.iloc[0]["equity"] == config.backtest.initial_cash
def test_run_simple_backtest_returns_empty_result_without_eligible_signals():
    config = load_config("configs/default.yaml")
    prices = sample_price_history(days=40)
    signals = pd.DataFrame(
        [
            {
                "date": prices.iloc[0]["date"],
                "code": "000001",
                "suggested_position": 0.15,
                "can_buy": False,
            }
        ]
    )

    equity, trades, metrics = run_simple_backtest(signals, prices, config.backtest)

    assert equity.empty
    assert trades.empty
    assert metrics == {
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "payoff_ratio": 0.0,
        "trade_count": 0,
        "avg_holding_days": 0.0,
    }


def test_payoff_ratio_is_finite_for_one_sided_and_flat_trade_samples():
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2),
            "equity": [100.0, 90.0],
        }
    )

    for returns in ([-0.1, -0.02], [0.1, 0.02], [0.0, 0.0]):
        trades = pd.DataFrame(
            {
                "return": returns,
                "holding_days": [2, 3],
            }
        )
        metrics = calculate_backtest_metrics(equity, trades)

        assert math.isfinite(metrics["payoff_ratio"])
        assert metrics["payoff_ratio"] == 0.0
        assert "nan" not in build_backtest_summary(metrics).lower()
        assert "inf" not in build_backtest_summary(metrics).lower()


def test_backtest_summary_explicitly_disclaims_historical_strategy_evidence():
    metrics = {
        "total_return": -0.0004,
        "max_drawdown": -0.0004,
        "win_rate": 0.0,
        "payoff_ratio": 0.0,
        "trade_count": 1,
        "avg_holding_days": 1.0,
    }

    summary = build_backtest_summary(
        metrics,
        data_mode="synthetic_demo",
        latest_data_date="2026-07-29",
    )

    assert "# 基础回测摘要（烟雾测试）" in summary
    assert "SYNTHETIC DEMO DATA / 合成演示数据" in summary
    assert "同日报告链路" in summary
    assert "不是历史多日期策略回测证据" in summary
