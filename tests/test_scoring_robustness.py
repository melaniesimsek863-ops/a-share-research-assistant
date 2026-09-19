from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from a_share_ai.diagnostics.robustness import (
    build_scoring_robustness_diagnostics,
    build_scoring_robustness_report,
    write_scoring_robustness_outputs,
)
from a_share_ai.models import StrategyConfig


def _robustness_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-29",
                "code": "000001",
                "tier": "focused",
                "medium_term_score": 90.0,
                "timing_score": 65.0,
                "total_score": 80.0,
                "signal_labels": "放量且收盘在20日均线上方,20日涨幅低于18%",
                "return_20d": 0.08,
                "return_60d": 0.18,
                "drawdown_60d": -0.07,
                "volume_ratio_5_20": 1.8,
                "above_ma20": True,
                "ma_bullish": True,
                "risk_penalty": 0.0,
                "forward_return_1d": 0.03,
                "forward_return_3d": 0.04,
            },
            {
                "date": "2026-07-29",
                "code": "000002",
                "tier": "candidate",
                "medium_term_score": 65.0,
                "timing_score": 30.0,
                "total_score": 50.0,
                "signal_labels": "pullback",
                "return_20d": 0.35,
                "return_60d": 0.60,
                "drawdown_60d": -0.02,
                "volume_ratio_5_20": 0.9,
                "above_ma20": True,
                "ma_bullish": True,
                "risk_penalty": 0.0,
                "forward_return_1d": -0.01,
                "forward_return_3d": -0.02,
            },
            {
                "date": "2026-07-30",
                "code": "000003",
                "tier": "focused",
                "medium_term_score": 88.0,
                "timing_score": 62.0,
                "total_score": 77.0,
                "signal_labels": "放量且收盘在20日均线上方",
                "return_20d": 0.32,
                "return_60d": 0.58,
                "drawdown_60d": -0.01,
                "volume_ratio_5_20": 1.5,
                "above_ma20": True,
                "ma_bullish": True,
                "risk_penalty": 0.0,
                "forward_return_1d": -0.04,
                "forward_return_3d": -0.05,
            },
            {
                "date": "2026-07-30",
                "code": "000004",
                "tier": "candidate",
                "medium_term_score": 70.0,
                "timing_score": 30.0,
                "total_score": 54.0,
                "signal_labels": "收盘在20日均线上方且60日回撤大于-8%,20日涨幅低于18%",
                "return_20d": 0.08,
                "return_60d": 0.18,
                "drawdown_60d": -0.07,
                "volume_ratio_5_20": 1.0,
                "above_ma20": True,
                "ma_bullish": True,
                "risk_penalty": 0.0,
                "forward_return_1d": 0.02,
                "forward_return_3d": 0.03,
            },
        ]
    )


def test_build_scoring_robustness_diagnostics_compares_dates_and_horizons() -> None:
    evaluated = _robustness_sample()
    original = evaluated.copy(deep=True)

    diagnostics = build_scoring_robustness_diagnostics(
        evaluated,
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    pd.testing.assert_frame_equal(evaluated, original)
    assert set(diagnostics) == {
        "by_date",
        "by_horizon",
        "quantile_monotonicity",
        "focused_overlap",
        "flags",
    }
    by_date = diagnostics["by_date"].sort_values(["date", "horizon"]).reset_index(drop=True)
    assert by_date[["date", "horizon"]].to_dict("records") == [
        {"date": "2026-07-29", "horizon": 1},
        {"date": "2026-07-29", "horizon": 3},
        {"date": "2026-07-30", "horizon": 1},
        {"date": "2026-07-30", "horizon": 3},
    ]
    first = by_date.iloc[0]
    assert first["production_focused_minus_candidate_avg_return"] == pytest.approx(0.04)
    assert first["experimental_focused_minus_candidate_avg_return"] == pytest.approx(0.04)
    assert "research_notice" in by_date.columns
    assert "recommendation_notice" in by_date.columns
    assert "scope_notice" in by_date.columns

    by_horizon = diagnostics["by_horizon"].sort_values("horizon").reset_index(drop=True)
    assert by_horizon.loc[0, "date_count"] == 2
    assert by_horizon.loc[0, "evaluable_date_count"] == 2
    assert by_horizon.loc[0, "avg_comparable_date_count"] == 2
    assert by_horizon.loc[0, "win_comparable_date_count"] == 2
    assert by_horizon.loc[0, "experimental_better_avg_date_rate"] >= 0.0
    assert by_horizon.loc[0, "experimental_better_win_date_rate"] >= 0.0


def test_incomplete_comparisons_are_excluded_from_improvement_rate_denominator() -> None:
    evaluated = _robustness_sample()
    evaluated.loc[evaluated["code"].eq("000004"), "forward_return_1d"] = float("nan")

    diagnostics = build_scoring_robustness_diagnostics(
        evaluated,
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    incomplete = diagnostics["by_date"].loc[
        diagnostics["by_date"]["date"].eq("2026-07-30")
        & diagnostics["by_date"]["horizon"].eq(1)
    ].iloc[0]
    summary = diagnostics["by_horizon"].loc[
        diagnostics["by_horizon"]["horizon"].eq(1)
    ].iloc[0]

    assert pd.isna(incomplete["avg_delta_improvement"])
    assert pd.isna(incomplete["experimental_better_avg"])
    assert summary["evaluable_date_count"] == 2
    assert summary["avg_comparable_date_count"] == 1
    assert summary["date_count"] == 1
    assert summary["experimental_better_avg_date_rate"] == pytest.approx(0.0)


def test_quantile_buckets_use_complete_date_level_scored_universe() -> None:
    evaluated = pd.concat([_robustness_sample().iloc[[0]].copy()] * 4, ignore_index=True)
    evaluated["code"] = ["000001", "000002", "000003", "000004"]
    evaluated["total_score"] = [10.0, 20.0, 30.0, 40.0]
    evaluated["forward_return_1d"] = [0.01, 0.02, 0.03, float("nan")]
    evaluated["forward_return_3d"] = [0.01, 0.02, 0.03, float("nan")]

    diagnostics = build_scoring_robustness_diagnostics(
        evaluated,
        StrategyConfig(focused_count=1, candidate_count=2),
    )
    production = diagnostics["quantile_monotonicity"].loc[
        diagnostics["quantile_monotonicity"]["score_type"].eq(
            "production_total_score"
        )
        & diagnostics["quantile_monotonicity"]["horizon"].eq(1)
    ].iloc[0]

    assert production["q3_avg_return"] == pytest.approx(0.03)
    assert pd.isna(production["q4_avg_return"])


def test_robustness_report_contains_auditable_sections_and_weak_stage_judgment() -> None:
    diagnostics = build_scoring_robustness_diagnostics(
        _robustness_sample().copy(),
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    report = build_scoring_robustness_report(diagnostics, "sample.csv")

    assert "# 多日期评分稳健性诊断报告" in report
    assert "ROBUSTNESS DIAGNOSTICS / 稳健性诊断" in report
    assert "HISTORICAL RESEARCH ONLY / 历史研究用途" in report
    assert "NOT A RECOMMENDATION / 不构成投资建议" in report
    assert "LIMITED CACHED REPLAY / 有限缓存池回放" in report
    assert "input replay date count: 2" in report
    assert "evaluable replay date count: 2" in report
    assert "## 最差日期" in report
    assert "## 分位组单调性" in report
    assert "## Focused 股票池重合" in report
    assert "低重合日期" in report
    assert "不支持进入 v0.6.6 权重校准" in report
    assert "不构成投资建议" in report


def test_robustness_flags_include_quantile_reversal() -> None:
    diagnostics = build_scoring_robustness_diagnostics(
        _robustness_sample(),
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    flags = diagnostics["flags"]

    assert set(flags["flag_type"]).issuperset({"quantile_reversal"})
    assert "research_notice" in flags.columns


def test_write_scoring_robustness_outputs_writes_all_auditable_csvs(tmp_path: Path) -> None:
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    _robustness_sample().to_csv(evaluated_path, index=False)

    paths = write_scoring_robustness_outputs(
        evaluated_path,
        tmp_path / "robustness",
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    assert paths == {
        "summary": tmp_path / "robustness" / "2026-07-30_robustness_summary.md",
        "by_date": tmp_path / "robustness" / "2026-07-30_robustness_by_date.csv",
        "by_horizon": tmp_path / "robustness" / "2026-07-30_robustness_by_horizon.csv",
        "quantile_monotonicity": (
            tmp_path
            / "robustness"
            / "2026-07-30_robustness_quantile_monotonicity.csv"
        ),
        "focused_overlap": (
            tmp_path / "robustness" / "2026-07-30_robustness_focused_overlap.csv"
        ),
        "flags": tmp_path / "robustness" / "2026-07-30_robustness_flags.csv",
    }
    assert paths["summary"].read_text(encoding="utf-8").startswith(
        "# 多日期评分稳健性诊断报告"
    )
    safety_columns = {
        "research_notice",
        "recommendation_notice",
        "scope_notice",
    }
    for key in (
        "by_date",
        "by_horizon",
        "quantile_monotonicity",
        "focused_overlap",
        "flags",
    ):
        assert safety_columns.issubset(pd.read_csv(paths[key]).columns)


def test_robustness_rejects_missing_required_columns() -> None:
    evaluated = _robustness_sample().drop(columns=["tier", "total_score"])

    with pytest.raises(ValueError, match="Missing required scoring robustness columns") as exc_info:
        build_scoring_robustness_diagnostics(
            evaluated,
            StrategyConfig(focused_count=1, candidate_count=2),
        )
    message = str(exc_info.value)
    assert "tier" in message
    assert "total_score" in message


def test_scoring_robustness_cli_writes_outputs(tmp_path: Path) -> None:
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    output_dir = tmp_path / "cli-output"
    _robustness_sample().to_csv(evaluated_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/scoring_robustness.py",
            "--evaluated",
            str(evaluated_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "robustness report: HISTORICAL RESEARCH ONLY / NOT A RECOMMENDATION" in completed.stdout
    assert (output_dir / "2026-07-30_robustness_summary.md").is_file()
    assert (output_dir / "2026-07-30_robustness_by_date.csv").is_file()
    assert (output_dir / "2026-07-30_robustness_by_horizon.csv").is_file()
    assert (output_dir / "2026-07-30_robustness_quantile_monotonicity.csv").is_file()
    assert (output_dir / "2026-07-30_robustness_focused_overlap.csv").is_file()
    assert (output_dir / "2026-07-30_robustness_flags.csv").is_file()