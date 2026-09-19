import math

import pandas as pd

from a_share_ai.backtest.replay import (
    ReplayConfig,
    evaluate_forward_returns,
    run_historical_replay,
    summarize_replay_results,
)
from a_share_ai.reports.replay_report import build_replay_summary


def _price_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                ]
            ),
            "code": ["000001"] * 6,
            "close": [10.0, 11.0, 9.0, 12.0, 8.0, 13.0],
        }
    )


def test_evaluate_forward_returns_uses_future_trading_days_after_candidate_date():
    # Catches: treating the candidate date as forward day 1, or using calendar-day offsets.
    candidates = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "rank": 1,
                "score": 88.0,
            }
        ]
    )

    evaluated = evaluate_forward_returns(candidates, _price_history(), horizons=(1, 3))

    row = evaluated.iloc[0]
    assert row["entry_date"] == pd.Timestamp("2026-01-02")
    assert row["forward_date_1d"] == pd.Timestamp("2026-01-05")
    assert row["forward_return_1d"] == 0.1
    assert row["forward_date_3d"] == pd.Timestamp("2026-01-07")
    assert row["forward_return_3d"] == 0.2
    assert row["has_forward_1d"] is True
    assert row["has_forward_3d"] is True


def test_evaluate_forward_returns_keeps_candidates_when_future_history_is_insufficient():
    # Catches: silently dropping candidates with insufficient future data.
    candidates = pd.DataFrame([{"date": "2026-01-08", "code": "000001"}])

    evaluated = evaluate_forward_returns(candidates, _price_history(), horizons=(1, 3))

    row = evaluated.iloc[0]
    assert row["entry_date"] == pd.Timestamp("2026-01-08")
    assert row["forward_date_1d"] == pd.Timestamp("2026-01-09")
    assert row["forward_return_1d"] == 0.625
    assert row["has_forward_1d"] is True
    assert pd.isna(row["forward_date_3d"])
    assert math.isnan(row["forward_return_3d"])
    assert row["has_forward_3d"] is False


def test_evaluate_forward_returns_leaves_missing_entry_when_no_price_on_or_after_candidate_date():
    # Catches: backfilling from prices before the candidate date, which would leak stale history.
    candidates = pd.DataFrame([{"date": "2026-01-12", "code": "000001"}])

    evaluated = evaluate_forward_returns(candidates, _price_history(), horizons=(1,))

    row = evaluated.iloc[0]
    assert pd.isna(row["entry_date"])
    assert math.isnan(row["entry_close"])
    assert row["has_forward_1d"] is False
    assert math.isnan(row["forward_return_1d"])


def test_summarize_replay_results_aggregates_each_horizon_independently():
    # Catches: counting unavailable horizons, or computing win rate from all candidates.
    evaluated = pd.DataFrame(
        {
            "code": ["000001", "000002", "000003"],
            "forward_return_1d": [0.1, -0.05, 0.0],
            "has_forward_1d": [True, True, True],
            "forward_return_3d": [0.2, float("nan"), -0.1],
            "has_forward_3d": [True, False, True],
        }
    )

    summary = summarize_replay_results(evaluated, horizons=(1, 3))

    assert summary.loc[summary["horizon"] == 1, "sample_count"].iloc[0] == 3
    assert summary.loc[summary["horizon"] == 1, "avg_return"].iloc[0] == 0.0167
    assert summary.loc[summary["horizon"] == 1, "median_return"].iloc[0] == 0.0
    assert summary.loc[summary["horizon"] == 1, "win_rate"].iloc[0] == 0.3333
    assert summary.loc[summary["horizon"] == 3, "sample_count"].iloc[0] == 2
    assert summary.loc[summary["horizon"] == 3, "avg_return"].iloc[0] == 0.05
    assert summary.loc[summary["horizon"] == 3, "median_return"].iloc[0] == 0.05
    assert summary.loc[summary["horizon"] == 3, "win_rate"].iloc[0] == 0.5


def test_run_historical_replay_returns_evaluated_candidates_and_summary():
    # Catches: orchestration returning only raw rows or only summary, making audit impossible.
    candidates = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "000001"},
            {"date": "2026-01-08", "code": "000001"},
        ]
    )
    config = ReplayConfig(horizons=(1, 3))

    evaluated, summary = run_historical_replay(candidates, _price_history(), config)

    assert len(evaluated) == 2
    assert list(summary["horizon"]) == [1, 3]
    assert summary.loc[summary["horizon"] == 1, "sample_count"].iloc[0] == 2
    assert summary.loc[summary["horizon"] == 3, "sample_count"].iloc[0] == 1


def test_replay_summary_explicitly_disclaims_investment_recommendations():
    # Catches: report wording that could be mistaken for live trading advice.
    summary = pd.DataFrame(
        {
            "horizon": [1, 3],
            "sample_count": [2, 1],
            "avg_return": [0.3625, 0.2],
            "median_return": [0.3625, 0.2],
            "win_rate": [1.0, 1.0],
        }
    )

    text = build_replay_summary(
        summary,
        candidate_count=2,
        replay_dates=("2026-01-02", "2026-01-08"),
    )

    assert "# 历史回放验证摘要" in text
    assert "HISTORICAL RESEARCH ONLY / 历史研究用途" in text
    assert "NOT A RECOMMENDATION / 不构成投资建议" in text
    assert "不代表未来收益" in text
    assert "| 1 | 2 | 36.25% | 36.25% | 100.00% |" in text
