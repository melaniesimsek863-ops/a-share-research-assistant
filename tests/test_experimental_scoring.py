import pandas as pd

from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import score_experimental_layer


def _experimental_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "000001",
                "tier": "focused",
                "medium_term_score": 98.0,
                "timing_score": 90.0,
                "total_score": 85.8,
                "risk_penalty": 0.0,
                "signal_labels": "放量且收盘在20日均线上方,收盘在20日均线上方且60日回撤大于-8%,20日涨幅低于18%",
                "above_ma20": True,
                "ma_bullish": True,
                "return_20d": 0.12,
                "return_60d": 0.22,
                "drawdown_60d": -0.06,
                "volume_ratio_5_20": 1.7,
            },
            {
                "code": "000002",
                "tier": "candidate",
                "medium_term_score": 88.0,
                "timing_score": 90.0,
                "total_score": 79.8,
                "risk_penalty": 0.0,
                "signal_labels": "放量且收盘在20日均线上方,收盘在20日均线上方且60日回撤大于-8%,20日涨幅低于18%",
                "above_ma20": True,
                "ma_bullish": True,
                "return_20d": 0.32,
                "return_60d": 0.58,
                "drawdown_60d": -0.02,
                "volume_ratio_5_20": 2.4,
            },
            {
                "code": "000003",
                "tier": "candidate",
                "medium_term_score": 65.0,
                "timing_score": 30.0,
                "total_score": 48.0,
                "risk_penalty": 0.0,
                "signal_labels": "收盘在20日均线上方且60日回撤大于-8%",
                "above_ma20": True,
                "ma_bullish": False,
                "return_20d": 0.04,
                "return_60d": 0.08,
                "drawdown_60d": -0.05,
                "volume_ratio_5_20": 1.0,
            },
        ]
    )


def test_experimental_scoring_appends_research_fields_without_changing_production_fields() -> None:
    rows = _experimental_rows()
    original = rows.copy(deep=True)

    scored = score_experimental_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))

    assert rows.equals(original)
    for column in ["medium_term_score", "timing_score", "total_score", "tier"]:
        assert scored[column].tolist() == original[column].tolist()
    assert list(scored.columns) == list(original.columns) + [
        "experimental_medium_term_score",
        "experimental_timing_score",
        "experimental_total_score",
        "experimental_rank",
        "experimental_tier",
    ]


def test_experimental_scoring_preserves_schema_without_risk_penalty() -> None:
    rows = _experimental_rows().drop(columns=["risk_penalty"])
    original = rows.copy(deep=True)

    scored = score_experimental_layer(rows, StrategyConfig(focused_count=1, candidate_count=2))

    assert rows.equals(original)
    assert list(scored.columns) == list(original.columns) + [
        "experimental_medium_term_score",
        "experimental_timing_score",
        "experimental_total_score",
        "experimental_rank",
        "experimental_tier",
    ]


def test_experimental_scoring_preserves_nonzero_risk_penalty_values() -> None:
    rows = _experimental_rows()
    rows["risk_penalty"] = [7.0, 3.0, 0.0]

    scored = score_experimental_layer(
        rows,
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    assert scored["risk_penalty"].tolist() == [7.0, 3.0, 0.0]
    assert list(scored.columns) == list(rows.columns) + [
        "experimental_medium_term_score",
        "experimental_timing_score",
        "experimental_total_score",
        "experimental_rank",
        "experimental_tier",
    ]


def test_experimental_scoring_penalizes_overheated_momentum() -> None:
    scored = score_experimental_layer(
        _experimental_rows(),
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    moderate = scored.loc[scored["code"].eq("000001")].iloc[0]
    overheated = scored.loc[scored["code"].eq("000002")].iloc[0]
    assert moderate["experimental_total_score"] > overheated["experimental_total_score"]
    assert overheated["experimental_tier"] != "focused"


def test_experimental_scoring_downweights_loose_drawdown_label() -> None:
    scored = score_experimental_layer(
        _experimental_rows(),
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    loose_only = scored.loc[scored["code"].eq("000003")].iloc[0]
    assert loose_only["experimental_timing_score"] == 15.0
    assert loose_only["experimental_total_score"] < 50.0

def test_experimental_scoring_restarts_rank_and_tier_for_each_replay_date() -> None:
    # Catches: globally ranking a multi-date replay, which leaves earlier dates
    # without their configured focused and candidate tiers.
    first_date = _experimental_rows().assign(date="2026-07-29")
    second_date = _experimental_rows().assign(
        date="2026-07-30", code=["000004", "000005", "000006"]
    )

    scored = score_experimental_layer(
        pd.concat([first_date, second_date], ignore_index=True),
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    for _, date_rows in scored.groupby("date"):
        assert sorted(date_rows["experimental_rank"].tolist()) == [1, 2, 3]
        assert (date_rows["experimental_tier"] == "focused").sum() == 1
        assert (date_rows["experimental_tier"] == "candidate").sum() == 1
        assert (date_rows["experimental_tier"] == "excluded").sum() == 1


def test_experimental_scoring_assigns_each_date_with_duplicate_input_index() -> None:
    # Catches: label-based tier assignment selecting matching duplicate labels
    # from another date, instead of only the rows in the current date group.
    first_date = _experimental_rows().assign(date="2026-07-29")
    second_date = _experimental_rows().assign(
        date="2026-07-30", code=["000004", "000005", "000006"]
    )
    rows = pd.concat([first_date, second_date])

    scored = score_experimental_layer(
        rows,
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    assert scored.index.equals(rows.index)
    for _, date_rows in scored.groupby("date"):
        assert sorted(date_rows["experimental_rank"].tolist()) == [1, 2, 3]
        assert (date_rows["experimental_tier"] == "focused").sum() == 1
        assert (date_rows["experimental_tier"] == "candidate").sum() == 1
        assert (date_rows["experimental_tier"] == "excluded").sum() == 1
