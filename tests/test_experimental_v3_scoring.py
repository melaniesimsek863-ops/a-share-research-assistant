import pandas as pd
import pytest

from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import (
    EXPERIMENTAL_V2_COLUMNS,
    EXPERIMENTAL_V3_COLUMNS,
    score_experimental_layer,
    score_experimental_v2_layer,
    score_experimental_v3_layer,
)


def _v3_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-29",
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
                "volume_ratio_5_20": 1.4,
            },
            {
                "date": "2026-07-29",
                "code": "000002",
                "tier": "candidate",
                "medium_term_score": 90.0,
                "timing_score": 86.0,
                "total_score": 84.0,
                "risk_penalty": 0.0,
                "signal_labels": "放量且收盘在20日均线上方",
                "above_ma20": True,
                "ma_bullish": True,
                "return_20d": 0.32,
                "return_60d": 0.58,
                "drawdown_60d": -0.02,
                "volume_ratio_5_20": 3.4,
            },
            {
                "date": "2026-07-29",
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


def test_experimental_v3_appends_fields_without_changing_existing_scores() -> None:
    rows = _v3_rows()
    v1 = score_experimental_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))
    v2 = score_experimental_v2_layer(v1, StrategyConfig(focused_count=1, candidate_count=2))
    original = v2.copy(deep=True)

    scored = score_experimental_v3_layer(v2, StrategyConfig(focused_count=1, candidate_count=2))

    preserved_columns = [
        "medium_term_score",
        "timing_score",
        "total_score",
        "tier",
        "experimental_total_score",
        "experimental_tier",
    ] + EXPERIMENTAL_V2_COLUMNS
    for column in preserved_columns:
        assert scored[column].tolist() == original[column].tolist()
    assert list(scored.columns) == list(original.columns) + EXPERIMENTAL_V3_COLUMNS


def test_experimental_v3_penalizes_overheated_setup_more_than_moderate_setup() -> None:
    scored = score_experimental_v3_layer(
        _v3_rows(), StrategyConfig(focused_count=1, candidate_count=2)
    )

    moderate = scored.loc[scored["code"].eq("000001")].iloc[0]
    overheated = scored.loc[scored["code"].eq("000002")].iloc[0]
    weak = scored.loc[scored["code"].eq("000003")].iloc[0]
    assert moderate["experimental_v3_total_score"] > overheated["experimental_v3_total_score"]
    assert moderate["experimental_v3_total_score"] > weak["experimental_v3_total_score"]
    assert overheated["experimental_v3_overheat_penalty_score"] > moderate["experimental_v3_overheat_penalty_score"]
    assert moderate["experimental_v3_anti_overheat_score"] > overheated["experimental_v3_anti_overheat_score"]


def test_experimental_v3_negative_20d_return_receives_no_extra_anti_overheat_bonus() -> None:
    rows = _v3_rows()
    rows.loc[rows["code"].eq("000003"), ["return_60d", "volume_ratio_5_20"]] = [0.18, 1.4]

    scored = score_experimental_v3_layer(
        rows, StrategyConfig(focused_count=1, candidate_count=2)
    )

    healthy = scored.loc[scored["code"].eq("000001")].iloc[0]
    negative = scored.loc[scored["code"].eq("000003")].iloc[0]
    assert negative["experimental_v3_anti_overheat_score"] == 55.0
    assert negative["experimental_v3_anti_overheat_score"] < healthy["experimental_v3_anti_overheat_score"]


def test_experimental_v3_restarts_rank_and_tier_for_each_replay_date() -> None:
    rows = pd.concat(
        [
            _v3_rows().assign(date="2026-07-29"),
            _v3_rows().assign(date="2026-07-30"),
        ],
        ignore_index=True,
    )

    scored = score_experimental_v3_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))

    for _, date_rows in scored.groupby("date"):
        assert (date_rows["experimental_v3_tier"] == "focused").sum() == 1
        assert (date_rows["experimental_v3_tier"] == "candidate").sum() == 1


def test_experimental_v3_preserves_duplicate_input_index() -> None:
    rows = _v3_rows()
    rows.index = [7, 7, 8]

    scored = score_experimental_v3_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))

    assert scored.index.equals(rows.index)


def test_experimental_v3_rejects_missing_required_columns() -> None:
    rows = _v3_rows().drop(columns=["return_20d", "above_ma20", "volume_ratio_5_20"])
    with pytest.raises(ValueError, match="Missing required experimental v3 scoring columns") as exc_info:
        score_experimental_v3_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))
    message = str(exc_info.value)
    assert "return_20d" in message
    assert "above_ma20" in message
    assert "volume_ratio_5_20" in message
