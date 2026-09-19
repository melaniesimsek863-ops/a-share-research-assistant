from pathlib import Path

import pandas as pd
import pytest

from a_share_ai.diagnostics.factor_effectiveness import (
    build_factor_effectiveness_diagnostics,
    build_factor_effectiveness_report,
    write_factor_effectiveness_outputs,
)


def _factor_sample() -> pd.DataFrame:
    rows = []
    for date_value in ["2026-07-29", "2026-07-30"]:
        for index, score in enumerate([10.0, 40.0, 70.0, 95.0], start=1):
            rows.append(
                {
                    "date": date_value,
                    "code": f"00000{index}",
                    "tier": "focused" if index == 4 else "candidate",
                    "medium_term_score": score,
                    "timing_score": 100.0 - score,
                    "total_score": score,
                    "experimental_total_score": 100.0 - score,
                    "signal_labels": "good_signal" if index >= 3 else "bad_signal",
                    "return_20d": score / 100,
                    "return_60d": score / 100,
                    "drawdown_60d": -0.20 + score / 500,
                    "volume_ratio_5_20": index,
                    "above_ma20": index >= 3,
                    "ma_bullish": index >= 3,
                    "forward_return_1d": (score - 50.0) / 1000,
                    "forward_return_3d": (score - 50.0) / 500,
                }
            )
    rows.append(
        {
            "date": "2026-07-30",
            "code": "000099",
            "tier": "candidate",
            "medium_term_score": 120.0,
            "timing_score": 0.0,
            "total_score": 120.0,
            "experimental_total_score": 0.0,
            "signal_labels": "missing_return",
            "return_20d": 1.2,
            "return_60d": 1.2,
            "drawdown_60d": 0.0,
            "volume_ratio_5_20": 9.0,
            "above_ma20": True,
            "ma_bullish": True,
            "forward_return_1d": pd.NA,
            "forward_return_3d": pd.NA,
        }
    )
    return pd.DataFrame(rows)


def test_factor_effectiveness_identifies_positive_and_negative_factors() -> None:
    diagnostics = build_factor_effectiveness_diagnostics(_factor_sample())
    by_factor = diagnostics["by_factor"]
    total = by_factor.loc[
        (by_factor["factor"].eq("total_score")) & (by_factor["horizon"].eq(1))
    ].iloc[0]
    inverse = by_factor.loc[
        (by_factor["factor"].eq("experimental_total_score"))
        & (by_factor["horizon"].eq(1))
    ].iloc[0]
    assert total["direction"] == "positive"
    assert total["mean_spearman_corr"] > 0
    assert inverse["direction"] == "negative"
    assert inverse["mean_spearman_corr"] < 0
    assert {"research_notice", "recommendation_notice", "scope_notice"}.issubset(
        by_factor.columns
    )


def test_factor_effectiveness_uses_date_level_buckets_before_return_drop() -> None:
    diagnostics = build_factor_effectiveness_diagnostics(_factor_sample())
    by_date = diagnostics["by_date"]
    top_bucket = by_date.loc[
        (by_date["date"].eq("2026-07-30"))
        & (by_date["factor"].eq("total_score"))
        & (by_date["bucket"].eq("q4"))
        & (by_date["horizon"].eq(1))
    ].iloc[0]
    assert top_bucket["sample_count"] == 0
    assert pd.isna(top_bucket["avg_return"])


def test_factor_effectiveness_writer_and_report_include_safety_copy(tmp_path: Path) -> None:
    assert "FACTOR EFFECTIVENESS / 因子有效性诊断" in build_factor_effectiveness_report(
        build_factor_effectiveness_diagnostics(_factor_sample()), "sample.csv"
    )
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    _factor_sample().to_csv(evaluated_path, index=False)
    paths = write_factor_effectiveness_outputs(evaluated_path, tmp_path / "factor")
    assert set(paths) == {"report", "by_factor", "by_date"}
    report = paths["report"].read_text(encoding="utf-8")
    assert "FACTOR EFFECTIVENESS / 因子有效性诊断" in report
    assert "HISTORICAL RESEARCH ONLY / 历史研究用途" in report
    assert "NOT A RECOMMENDATION / 不构成投资建议" in report
    assert "LIMITED CACHED REPLAY / 有限缓存池回放" in report
    for key in ("by_factor", "by_date"):
        assert {
            "research_notice",
            "recommendation_notice",
            "scope_notice",
        }.issubset(pd.read_csv(paths[key]).columns)


def test_factor_effectiveness_rejects_missing_required_columns() -> None:
    with pytest.raises(
        ValueError, match="Missing required factor effectiveness columns"
    ) as exc_info:
        build_factor_effectiveness_diagnostics(
            _factor_sample().drop(columns=["date", "total_score"])
        )
    message = str(exc_info.value)
    assert "date" in message
    assert "total_score" in message


def test_factor_effectiveness_joins_q1_q4_by_date() -> None:
    sample = _factor_sample()
    sample.loc[(sample["date"] == "2026-07-29") & (sample["code"] == "000001"), "forward_return_1d"] = -0.5
    sample = sample.loc[~((sample["date"] == "2026-07-29") & (sample["code"] == "000004"))].reset_index(drop=True)
    sample.loc[sample["code"] == "000099", "forward_return_1d"] = 0.08
    diagnostics = build_factor_effectiveness_diagnostics(sample)
    row = diagnostics["by_factor"].loc[
        (diagnostics["by_factor"]["factor"] == "total_score")
        & (diagnostics["by_factor"]["horizon"] == 1)
    ].iloc[0]
    assert row["q4_minus_q1_mean"] == pytest.approx(0.3125)
    assert row["q4_minus_q1_positive_rate"] == 1.0



def test_factor_effectiveness_binary_factor_uses_q1_q4_direction() -> None:
    diagnostics = build_factor_effectiveness_diagnostics(_factor_sample())
    row = diagnostics["by_factor"].loc[
        (diagnostics["by_factor"]["factor"] == "above_ma20")
        & (diagnostics["by_factor"]["horizon"] == 1)
    ].iloc[0]
    assert row["direction"] == "positive"
    assert row["q4_minus_q1_mean"] > 0

@pytest.mark.parametrize("missing", ["above_ma20", "ma_bullish", "experimental_total_score"])
def test_factor_effectiveness_rejects_missing_factor_columns(missing: str) -> None:
    with pytest.raises(ValueError, match="Missing required factor effectiveness columns") as exc_info:
        build_factor_effectiveness_diagnostics(_factor_sample().drop(columns=[missing]))
    assert missing in str(exc_info.value)
