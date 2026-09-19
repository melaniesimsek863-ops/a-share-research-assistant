from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from a_share_ai.diagnostics.scoring import (
    build_experimental_scoring_diagnostics,
    build_scoring_diagnostics,
    write_scoring_diagnostics_outputs,
)
from a_share_ai.models import StrategyConfig


def _sample_evaluated() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-29",
                "code": "000001",
                "tier": "focused",
                "medium_term_score": 90.0,
                "timing_score": 60.0,
                "total_score": 72.0,
                "signal_labels": "breakout,strong_trend",
                "return_20d": 0.12,
                "return_60d": 0.25,
                "drawdown_60d": -0.04,
                "volume_ratio_5_20": 1.8,
                "forward_return_1d": -0.04,
                "forward_return_3d": -0.10,
            },
            {
                "date": "2026-07-29",
                "code": "000002",
                "tier": "candidate",
                "medium_term_score": 70.0,
                "timing_score": 30.0,
                "total_score": 51.0,
                "signal_labels": "pullback",
                "return_20d": 0.03,
                "return_60d": 0.06,
                "drawdown_60d": -0.12,
                "volume_ratio_5_20": 0.9,
                "forward_return_1d": 0.02,
                "forward_return_3d": 0.03,
            },
            {
                "date": "2026-07-30",
                "code": "000003",
                "tier": "focused",
                "medium_term_score": 85.0,
                "timing_score": 60.0,
                "total_score": 69.0,
                "signal_labels": "strong_trend",
                "return_20d": 0.09,
                "return_60d": 0.18,
                "drawdown_60d": -0.03,
                "volume_ratio_5_20": 1.2,
                "forward_return_1d": -0.02,
                "forward_return_3d": pd.NA,
            },
            {
                "date": "2026-07-30",
                "code": "000004",
                "tier": "candidate",
                "medium_term_score": 65.0,
                "timing_score": 30.0,
                "total_score": 48.0,
                "signal_labels": "pullback",
                "return_20d": -0.02,
                "return_60d": 0.01,
                "drawdown_60d": -0.18,
                "volume_ratio_5_20": 0.7,
                "forward_return_1d": 0.04,
                "forward_return_3d": 0.05,
            },
        ]
    )


def test_experimental_scoring_diagnostics_compare_research_tier_without_changing_production_tier() -> None:
    # Catches: experimental ranking overwriting production fields, calculating tier delta
    # from production tiers, or reporting focused-set overlap from the wrong tier.
    evaluated = _sample_evaluated().assign(
        above_ma20=[True, True, True, False],
        ma_bullish=[True, True, True, False],
        risk_penalty=[0.0, 0.0, 0.0, 0.0],
    )
    original = evaluated.copy(deep=True)

    diagnostics = build_experimental_scoring_diagnostics(
        evaluated,
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    assert set(diagnostics) == {
        "experimental_tier_delta",
        "experimental_factor_buckets",
        "experimental_overlap",
    }
    pd.testing.assert_frame_equal(evaluated, original)
    overlap = diagnostics["experimental_overlap"]
    assert set(overlap.columns) == {
        "date",
        "production_focused_count",
        "experimental_focused_count",
        "focused_overlap_count",
        "focused_overlap_rate",
    }
    assert overlap.sort_values("date").to_dict("records") == [
        {"date": "2026-07-29", "production_focused_count": 1, "experimental_focused_count": 1, "focused_overlap_count": 1, "focused_overlap_rate": 1.0},
        {"date": "2026-07-30", "production_focused_count": 1, "experimental_focused_count": 1, "focused_overlap_count": 1, "focused_overlap_rate": 1.0},
    ]
    assert diagnostics["experimental_tier_delta"].sort_values("horizon").to_dict("records") == [
        {"horizon": 1, "focused_sample_count": 2, "candidate_sample_count": 2, "focused_avg_return": -0.03, "candidate_avg_return": 0.03, "focused_median_return": -0.03, "candidate_median_return": 0.03, "focused_win_rate": 0.0, "candidate_win_rate": 1.0, "focused_minus_candidate_avg_return": -0.06, "focused_minus_candidate_win_rate": -1.0},
        {"horizon": 3, "focused_sample_count": 1, "candidate_sample_count": 2, "focused_avg_return": -0.10, "candidate_avg_return": 0.04, "focused_median_return": -0.10, "candidate_median_return": 0.04, "focused_win_rate": 0.0, "candidate_win_rate": 1.0, "focused_minus_candidate_avg_return": -0.14, "focused_minus_candidate_win_rate": -1.0},
    ]

def test_experimental_factor_buckets_use_experimental_total_score() -> None:
    # Catches: research diagnostics using the production total-score buckets by accident.
    evaluated = _sample_evaluated().assign(
        above_ma20=True,
        ma_bullish=True,
        risk_penalty=0.0,
    )

    diagnostics = build_experimental_scoring_diagnostics(
        evaluated,
        StrategyConfig(focused_count=1, candidate_count=2),
    )

    buckets = diagnostics["experimental_factor_buckets"]
    assert "experimental_total_score" in set(buckets["factor"])
    total_rows = buckets.loc[buckets["factor"].eq("experimental_total_score")]
    assert set(total_rows["horizon"]) == {1, 3}

def test_build_scoring_diagnostics_computes_focused_candidate_tier_delta() -> None:
    # Catches: tier delta hiding focused underperformance or counting missing horizons as samples.
    diagnostics = build_scoring_diagnostics(_sample_evaluated())

    tier_delta = diagnostics["tier_delta"].sort_values("horizon").reset_index(drop=True)
    assert tier_delta.to_dict("records") == [
        {
            "horizon": 1,
            "focused_sample_count": 2,
            "candidate_sample_count": 2,
            "focused_avg_return": -0.03,
            "candidate_avg_return": 0.03,
            "focused_median_return": -0.03,
            "candidate_median_return": 0.03,
            "focused_win_rate": 0.0,
            "candidate_win_rate": 1.0,
            "focused_minus_candidate_avg_return": -0.06,
            "focused_minus_candidate_win_rate": -1.0,
        },
        {
            "horizon": 3,
            "focused_sample_count": 1,
            "candidate_sample_count": 2,
            "focused_avg_return": -0.10,
            "candidate_avg_return": 0.04,
            "focused_median_return": -0.10,
            "candidate_median_return": 0.04,
            "focused_win_rate": 0.0,
            "candidate_win_rate": 1.0,
            "focused_minus_candidate_avg_return": -0.14,
            "focused_minus_candidate_win_rate": -1.0,
        },
    ]


def test_build_scoring_diagnostics_rejects_missing_required_columns() -> None:
    # Catches: diagnostics silently producing misleading output when score columns are absent.
    evaluated = _sample_evaluated().drop(columns=["timing_score", "volume_ratio_5_20"])

    with pytest.raises(ValueError, match="Missing required scoring diagnostics columns"):
        build_scoring_diagnostics(evaluated)


def test_build_scoring_diagnostics_returns_schema_complete_empty_tier_delta_when_returns_missing() -> None:
    evaluated = _sample_evaluated().assign(
        forward_return_1d=pd.NA,
        forward_return_3d=pd.NA,
    )

    tier_delta = build_scoring_diagnostics(evaluated)["tier_delta"]

    assert tier_delta.empty
    assert list(tier_delta.columns) == [
        "horizon",
        "focused_sample_count",
        "candidate_sample_count",
        "focused_avg_return",
        "candidate_avg_return",
        "focused_median_return",
        "candidate_median_return",
        "focused_win_rate",
        "candidate_win_rate",
        "focused_minus_candidate_avg_return",
        "focused_minus_candidate_win_rate",
    ]


def test_score_factor_buckets_include_quantile_total_score_buckets() -> None:
    # Catches: fixed score buckets hiding all samples in one low-score bucket.
    diagnostics = build_scoring_diagnostics(_sample_evaluated())

    buckets = diagnostics["score_factor_buckets"]
    total_quantile = buckets.loc[
        buckets["factor"].eq("total_score") & buckets["bucket_type"].eq("quantile")
    ]
    assert set(total_quantile["horizon"]) == {1, 3}
    horizon_one = total_quantile.loc[total_quantile["horizon"].eq(1)]
    assert horizon_one["bucket"].nunique() >= 2
    assert horizon_one["sample_count"].sum() == 4



def test_score_factor_buckets_keep_equal_total_scores_in_one_quantile_bucket() -> None:
    # Catches: row-order ranking splitting identical scores across artificial quantiles.
    evaluated = _sample_evaluated().assign(total_score=70.0)

    buckets = build_scoring_diagnostics(evaluated)["score_factor_buckets"]
    total_quantile = buckets.loc[
        buckets["factor"].eq("total_score") & buckets["bucket_type"].eq("quantile")
    ]

    assert set(total_quantile["bucket"]) == {"q1"}


def test_score_factor_bucket_membership_does_not_change_when_later_horizon_is_missing() -> None:
    # Catches: long-horizon expansion assigning one code's identical factor values to different buckets.
    evaluated = _sample_evaluated().iloc[:3].copy()
    evaluated["total_score"] = [10.0, 20.0, 30.0]
    evaluated.loc[evaluated["code"].eq("000003"), "forward_return_3d"] = pd.NA

    buckets = build_scoring_diagnostics(evaluated)["score_factor_buckets"]
    total_quantile = buckets.loc[
        buckets["factor"].eq("total_score") & buckets["bucket_type"].eq("quantile")
    ]
    horizon_one = total_quantile.loc[total_quantile["horizon"].eq(1)]
    horizon_three = total_quantile.loc[total_quantile["horizon"].eq(3)]

    assert set(horizon_one["bucket"]) == {"q1", "q2", "q3"}
    assert set(horizon_three["bucket"]) == {"q1", "q2"}


def test_signal_label_diagnostics_expand_labels_without_polluting_tier_delta() -> None:
    # Catches: multi-label rows inflating tier-level sample counts.
    diagnostics = build_scoring_diagnostics(_sample_evaluated())

    tier_delta = diagnostics["tier_delta"].loc[diagnostics["tier_delta"]["horizon"].eq(1)].iloc[0]
    assert tier_delta["focused_sample_count"] == 2
    labels = diagnostics["signal_label_diagnostics"]
    label_rows = labels.loc[labels["horizon"].eq(1)].sort_values("signal_label")
    assert label_rows[["signal_label", "sample_count"]].to_dict("records") == [
        {"signal_label": "breakout", "sample_count": 1},
        {"signal_label": "pullback", "sample_count": 2},
        {"signal_label": "strong_trend", "sample_count": 2},
    ]
    strong_trend = labels.loc[
        labels["signal_label"].eq("strong_trend") & labels["horizon"].eq(1)
    ].iloc[0]
    assert strong_trend["focused_share"] == 1.0


def test_signal_combo_diagnostics_preserve_original_label_combinations() -> None:
    # Catches: losing combo-level evidence by only looking at single labels.
    diagnostics = build_scoring_diagnostics(_sample_evaluated())

    combos = diagnostics["signal_combo_diagnostics"]
    combo_names = set(combos.loc[combos["horizon"].eq(1), "signal_combo"])
    assert combo_names == {"breakout | strong_trend", "pullback", "strong_trend"}


def test_bad_replay_dates_returns_worst_average_dates_by_horizon() -> None:
    # Catches: bad-date diagnostics sorting in the wrong direction.
    diagnostics = build_scoring_diagnostics(_sample_evaluated())

    bad_dates = diagnostics["bad_replay_dates"]
    horizon_one = bad_dates.loc[bad_dates["horizon"].eq(1)].reset_index(drop=True)
    assert horizon_one.iloc[0]["date"] == "2026-07-29"
    assert horizon_one.iloc[0]["avg_return"] == -0.01
def test_write_scoring_diagnostics_outputs_writes_report_and_all_csvs(tmp_path: Path) -> None:
    # Catches: diagnostics only existing in memory without auditable artifacts.
    case_dir = tmp_path / "writer"
    case_dir.mkdir()
    evaluated_path = case_dir / "2026-07-31_replay_evaluated.csv"
    _sample_evaluated().to_csv(evaluated_path, index=False)

    paths = write_scoring_diagnostics_outputs(evaluated_path, case_dir / "diagnostics")

    assert {key: path.name for key, path in paths.items()} == {
        "report": "2026-07-31_scoring_diagnostics.md",
        "tier_delta": "2026-07-31_tier_delta.csv",
        "score_factor_buckets": "2026-07-31_score_factor_buckets.csv",
        "signal_label_diagnostics": "2026-07-31_signal_label_diagnostics.csv",
        "signal_combo_diagnostics": "2026-07-31_signal_combo_diagnostics.csv",
        "bad_replay_dates": "2026-07-31_bad_replay_dates.csv",
    }
    assert all(path.is_file() for path in paths.values())
    report = paths["report"].read_text(encoding="utf-8")
    assert "SCORING DIAGNOSTICS / 评分诊断" in report
    assert "HISTORICAL RESEARCH ONLY / 历史研究用途" in report
    assert "NOT A RECOMMENDATION / 不构成投资建议" in report
    assert "LIMITED CACHED REPLAY / 有限缓存池回放" in report
    assert "## Factor buckets" in report
    assert "## Signal labels and combinations" in report
    assert "## Bad replay dates" in report



def test_write_scoring_diagnostics_outputs_can_write_experimental_comparison(tmp_path: Path) -> None:
    # Catches: opting into research diagnostics not producing auditable experimental artifacts.
    case_dir = tmp_path / "writer-experimental"
    case_dir.mkdir()
    evaluated_path = case_dir / "2026-07-31_replay_evaluated.csv"
    evaluated = _sample_evaluated().assign(above_ma20=True, ma_bullish=True, risk_penalty=0.0)
    evaluated.to_csv(evaluated_path, index=False)
    paths = write_scoring_diagnostics_outputs(evaluated_path, case_dir / "diagnostics", include_experimental=True, strategy_config=StrategyConfig(focused_count=1, candidate_count=2))
    assert paths["experimental_report"].name == "2026-07-31_experimental_scoring_comparison.md"
    assert paths["experimental_tier_delta"].name == "2026-07-31_experimental_tier_delta.csv"
    assert paths["experimental_factor_buckets"].name == "2026-07-31_experimental_factor_buckets.csv"
    assert paths["experimental_overlap"].name == "2026-07-31_experimental_overlap.csv"
    report = paths["experimental_report"].read_text(encoding="utf-8")
    assert "EXPERIMENTAL SCORING COMPARISON / 实验评分对比" in report
    assert "HISTORICAL RESEARCH ONLY / 历史研究用途" in report
    assert "NOT A RECOMMENDATION / 不构成投资建议" in report
    assert "does not change production tier" in report
def test_scoring_diagnostics_cli_runs_as_subprocess_and_reports_outputs(tmp_path: Path) -> None:
    # Catches: scoring diagnostics not being available as a user-runnable workflow.
    case_dir = tmp_path / "cli"
    case_dir.mkdir()
    evaluated_path = case_dir / "2026-07-31_replay_evaluated.csv"
    output_dir = case_dir / "diagnostics"
    _sample_evaluated().to_csv(evaluated_path, index=False)
    script = Path(__file__).parents[1] / "scripts" / "scoring_diagnostics.py"

    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            str(script),
            "--evaluated",
            str(evaluated_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SCORING DIAGNOSTICS / NOT A RECOMMENDATION" in completed.stdout
    assert "report path:" in completed.stdout
    assert (output_dir / "2026-07-31_scoring_diagnostics.md").is_file()
    assert (output_dir / "2026-07-31_signal_label_diagnostics.csv").is_file()
def test_scoring_diagnostics_cli_can_write_experimental_comparison(tmp_path: Path) -> None:
    # Catches: --experimental not loading config or omitting experimental artifacts.
    case_dir = tmp_path / "cli-experimental"
    case_dir.mkdir()
    evaluated_path = case_dir / "2026-07-31_replay_evaluated.csv"
    output_dir = case_dir / "diagnostics"
    evaluated = _sample_evaluated().assign(above_ma20=True, ma_bullish=True, risk_penalty=0.0)
    evaluated.to_csv(evaluated_path, index=False)
    script = Path(__file__).parents[1] / "scripts" / "scoring_diagnostics.py"
    completed = subprocess.run([str(Path(sys.executable)), str(script), "--evaluated", str(evaluated_path), "--output-dir", str(output_dir), "--experimental", "--config", str(Path(__file__).parents[1] / "configs" / "default.yaml")], cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "experimental scoring comparison" in completed.stdout
    assert (output_dir / "2026-07-31_experimental_scoring_comparison.md").is_file()
    assert (output_dir / "2026-07-31_experimental_overlap.csv").is_file()

def test_experimental_outputs_compare_production_and_include_csv_safety_metadata(tmp_path: Path) -> None:
    # Catches: a detached experimental CSV losing its research-only safety context.
    case_dir = tmp_path / "writer-experimental-review"
    case_dir.mkdir()
    evaluated_path = case_dir / "2026-07-31_replay_evaluated.csv"
    _sample_evaluated().assign(above_ma20=True, ma_bullish=True, risk_penalty=0.0).to_csv(evaluated_path, index=False)
    paths = write_scoring_diagnostics_outputs(evaluated_path, case_dir / "diagnostics", include_experimental=True, strategy_config=StrategyConfig(focused_count=1, candidate_count=2))

    report = paths["experimental_report"].read_text(encoding="utf-8")
    assert "## Production vs experimental tier delta" in report
    assert "Production delta avg" in report
    assert "## Production vs experimental total-score quantiles" in report
    assert "## Focused overlap by date" in report
    for key in ["experimental_tier_delta", "experimental_factor_buckets", "experimental_overlap"]:
        csv = pd.read_csv(paths[key])
        assert csv["research_notice"].eq("HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014").all()
        assert csv["recommendation_notice"].eq("NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae").all()
        assert csv["scope_notice"].eq("LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e").all()
