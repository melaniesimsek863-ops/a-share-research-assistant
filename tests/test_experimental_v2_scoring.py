import pandas as pd
import pytest

from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import (
    EXPERIMENTAL_V2_COLUMNS,
    score_experimental_layer,
    score_experimental_v2_layer,
)


def _v2_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "000001",
                "tier": "focused",
                "medium_term_score": 92.0,
                "timing_score": 88.0,
                "total_score": 87.0,
                "risk_penalty": 0.0,
                "signal_labels": "放量且收盘在20日均线上方,收盘在20日均线上方且60日回撤大于-8%,20日涨幅低于18%",
                "above_ma20": True,
                "ma_bullish": True,
                "return_20d": 0.08,
                "return_60d": 0.18,
                "drawdown_60d": -0.07,
                "volume_ratio_5_20": 1.5,
            },
            {
                "code": "000002",
                "tier": "candidate",
                "medium_term_score": 90.0,
                "timing_score": 86.0,
                "total_score": 84.0,
                "risk_penalty": 0.0,
                "signal_labels": "放量且收盘在20日均线上方,收盘在20日均线上方且60日回撤大于-8%",
                "above_ma20": True,
                "ma_bullish": True,
                "return_20d": 0.35,
                "return_60d": 0.60,
                "drawdown_60d": -0.02,
                "volume_ratio_5_20": 2.8,
            },
            {
                "code": "000003",
                "tier": "candidate",
                "medium_term_score": 55.0,
                "timing_score": 25.0,
                "total_score": 43.0,
                "risk_penalty": 0.0,
                "signal_labels": "",
                "above_ma20": False,
                "ma_bullish": False,
                "return_20d": -0.04,
                "return_60d": -0.08,
                "drawdown_60d": -0.22,
                "volume_ratio_5_20": 0.7,
            },
        ]
    )

def test_experimental_v2_appends_fields_without_changing_existing_scores() -> None:
    rows = _v2_rows()
    v1 = score_experimental_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))
    original = v1.copy(deep=True)

    scored = score_experimental_v2_layer(v1, StrategyConfig(focused_count=1, candidate_count=2))

    for column in [
        "medium_term_score",
        "timing_score",
        "total_score",
        "tier",
        "experimental_total_score",
        "experimental_tier",
    ]:
        assert scored[column].tolist() == original[column].tolist()
    assert list(scored.columns) == list(original.columns) + EXPERIMENTAL_V2_COLUMNS


def test_experimental_v2_penalizes_overheated_momentum_more_than_moderate_setup() -> None:
    scored = score_experimental_v2_layer(
        _v2_rows(), StrategyConfig(focused_count=1, candidate_count=2)
    )

    moderate = scored.loc[scored["code"].eq("000001")].iloc[0]
    overheated = scored.loc[scored["code"].eq("000002")].iloc[0]
    assert moderate["experimental_v2_total_score"] > overheated["experimental_v2_total_score"]
    assert overheated["experimental_v2_penalty_score"] > moderate["experimental_v2_penalty_score"]


def test_experimental_v2_restarts_rank_and_tier_for_each_replay_date() -> None:
    rows = pd.concat(
        [
            _v2_rows().assign(date="2026-07-29"),
            _v2_rows().assign(date="2026-07-30"),
        ],
        ignore_index=True,
    )

    scored = score_experimental_v2_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))

    for _, date_rows in scored.groupby("date"):
        assert (date_rows["experimental_v2_tier"] == "focused").sum() == 1
        assert (date_rows["experimental_v2_tier"] == "candidate").sum() == 1


def test_experimental_v2_preserves_duplicate_input_index() -> None:
    rows = _v2_rows()
    rows.index = [7, 7, 8]

    scored = score_experimental_v2_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))

    assert scored.index.equals(rows.index)

def test_experimental_v2_rejects_missing_required_columns() -> None:
    rows = _v2_rows().drop(columns=["return_20d", "above_ma20", "volume_ratio_5_20"])
    with pytest.raises(ValueError, match="Missing required experimental v2 scoring columns") as exc_info:
        score_experimental_v2_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))
    message = str(exc_info.value)
    assert "return_20d" in message
    assert "above_ma20" in message
    assert "volume_ratio_5_20" in message
