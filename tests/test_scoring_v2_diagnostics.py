from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from a_share_ai.diagnostics.robustness import (
    EXPERIMENTAL_V2_PROFILE,
    build_scoring_robustness_diagnostics_for_profile,
)
from a_share_ai.diagnostics.scoring_v2 import (
    build_v2_diagnostics_report,
    write_scoring_v2_diagnostics_outputs,
)
from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import score_experimental_v2_layer


def _v2_diagnostic_sample() -> pd.DataFrame:
    rows = []
    for date_value in ("2026-07-29", "2026-07-30"):
        for index, (return_20d, return_60d, forward_return) in enumerate(
            [
                (0.08, 0.18, 0.05),
                (0.05, 0.14, 0.02),
                (0.01, 0.04, -0.01),
                (-0.04, -0.08, -0.03),
            ],
            start=1,
        ):
            rows.append(
                {
                    "date": date_value,
                    "code": f"{date_value[-2:]}000{index}",
                    "tier": "focused" if index == 1 else "candidate",
                    "medium_term_score": 100.0 - index * 10,
                    "timing_score": 90.0 - index * 8,
                    "total_score": 95.0 - index * 12,
                    "signal_labels": (
                        "放量且收盘在20日均线上方,"
                        "收盘在20日均线上方且60日回撤大于-8%,20日涨幅低于18%"
                        if index == 1
                        else ""
                    ),
                    "return_20d": return_20d,
                    "return_60d": return_60d,
                    "drawdown_60d": -0.07 if index < 3 else -0.18,
                    "volume_ratio_5_20": 1.5 if index < 3 else 0.8,
                    "above_ma20": index < 3,
                    "ma_bullish": index < 3,
                    "risk_penalty": 0.0,
                    "forward_return_1d": forward_return,
                    "forward_return_3d": forward_return * 1.5,
                }
            )
    return pd.DataFrame(rows)


def _weak_v2_sample() -> pd.DataFrame:
    sample = _v2_diagnostic_sample()
    sample.loc[sample["code"].str.endswith("0001"), ["forward_return_1d", "forward_return_3d"]] = -0.05
    sample.loc[sample["code"].str.endswith("0002"), ["forward_return_1d", "forward_return_3d"]] = 0.05
    return sample


def _factor_diagnostics_stub() -> dict[str, pd.DataFrame]:
    return {
        "by_factor": pd.DataFrame(
            [
                {
                    "factor": "total_score",
                    "horizon": 1,
                    "direction": "positive",
                    "q4_minus_q1_mean": 0.03,
                    "mean_spearman_corr": 0.8,
                },
                {
                    "factor": "experimental_total_score",
                    "horizon": 1,
                    "direction": "negative",
                    "q4_minus_q1_mean": -0.02,
                    "mean_spearman_corr": -0.5,
                },
            ]
        )
    }


def _diagnostics_stub(horizons: list[int], *, improved_reversals: bool) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    v1_horizon_rows = []
    v2_horizon_rows = []
    v1_date_rows = []
    v2_date_rows = []
    v1_quantile_rows = []
    v2_quantile_rows = []
    for horizon in horizons:
        v1_horizon_rows.append(
            {
                "horizon": horizon,
                "experimental_overall_delta_avg": 0.01,
                "experimental_worst_delta_avg": -0.03,
            }
        )
        v2_horizon_rows.append(
            {
                "horizon": horizon,
                "experimental_overall_delta_avg": 0.02 if horizon in {1, 3, 5} else (0.011 if horizon == 20 else 0.0),
                "experimental_worst_delta_avg": -0.025,
            }
        )
        v1_date_rows.append(
            {
                "date": f"2026-07-{20 + horizon % 8:02d}",
                "horizon": horizon,
                "experimental_focused_minus_candidate_avg_return": -0.03,
            }
        )
        v2_date_rows.append(
            {
                "date": f"2026-07-{21 + horizon % 8:02d}",
                "horizon": horizon,
                "experimental_focused_minus_candidate_avg_return": -0.025,
            }
        )
        v1_quantile_rows.append(
            {
                "date": "2026-07-29",
                "horizon": horizon,
                "score_type": "experimental_total_score",
                "q1_avg_return": 0.03,
                "q4_avg_return": 0.01,
                "q4_below_q1": True,
                "q4_below_q3": False,
            }
        )
        v2_quantile_rows.append(
            {
                "date": "2026-07-29",
                "horizon": horizon,
                "score_type": "experimental_v2_total_score",
                "q1_avg_return": 0.01,
                "q4_avg_return": 0.03,
                "q4_below_q1": not improved_reversals,
                "q4_below_q3": False,
            }
        )
    return (
        {
            "by_horizon": pd.DataFrame(v1_horizon_rows),
            "by_date": pd.DataFrame(v1_date_rows),
            "quantile_monotonicity": pd.DataFrame(v1_quantile_rows),
        },
        {
            "by_horizon": pd.DataFrame(v2_horizon_rows),
            "by_date": pd.DataFrame(v2_date_rows),
            "quantile_monotonicity": pd.DataFrame(v2_quantile_rows),
        },
    )


def test_v2_profile_requires_v2_columns_and_adds_profile_name() -> None:
    with pytest.raises(ValueError, match="experimental_v2_total_score"):
        build_scoring_robustness_diagnostics_for_profile(
            _v2_diagnostic_sample(),
            StrategyConfig(focused_count=1, candidate_count=2),
            EXPERIMENTAL_V2_PROFILE,
        )
    scored = score_experimental_v2_layer(
        _v2_diagnostic_sample(), StrategyConfig(focused_count=1, candidate_count=2)
    )
    diagnostics = build_scoring_robustness_diagnostics_for_profile(
        scored,
        StrategyConfig(focused_count=1, candidate_count=2),
        EXPERIMENTAL_V2_PROFILE,
    )
    assert set(diagnostics["by_horizon"]["score_profile"]) == {"experimental_v2"}


def test_write_scoring_v2_diagnostics_outputs_creates_complete_artifacts(
    tmp_path: Path,
) -> None:
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    _v2_diagnostic_sample().to_csv(evaluated_path, index=False)
    output_dir = tmp_path / "v2"
    paths = write_scoring_v2_diagnostics_outputs(
        evaluated_path,
        output_dir,
        StrategyConfig(focused_count=1, candidate_count=2),
    )
    assert paths == {
        "factor_report": output_dir / "2026-07-30_factor_effectiveness_report.md",
        "factor_by_factor": output_dir / "2026-07-30_factor_effectiveness_by_factor.csv",
        "factor_by_date": output_dir / "2026-07-30_factor_effectiveness_by_date.csv",
        "v2_report": output_dir / "2026-07-30_v2_diagnostics_report.md",
        "v2_summary": output_dir / "2026-07-30_v2_robustness_summary.md",
        "v2_by_date": output_dir / "2026-07-30_v2_robustness_by_date.csv",
        "v2_by_horizon": output_dir / "2026-07-30_v2_robustness_by_horizon.csv",
        "v2_quantile_monotonicity": output_dir / "2026-07-30_v2_robustness_quantile_monotonicity.csv",
        "v2_focused_overlap": output_dir / "2026-07-30_v2_robustness_focused_overlap.csv",
        "v2_flags": output_dir / "2026-07-30_v2_robustness_flags.csv",
    }
    report = paths["v2_report"].read_text(encoding="utf-8")
    assert "V2 SCORING DIAGNOSTICS / 实验评分 v2 诊断" in report
    assert "v1 vs v2" in report
    assert "下一阶段判断" in report
    assert "不构成投资建议" in report
    assert "当前不支持进入权重校准" in paths["v2_summary"].read_text(encoding="utf-8")
    for key in (
        "factor_by_factor",
        "factor_by_date",
        "v2_by_date",
        "v2_by_horizon",
        "v2_quantile_monotonicity",
        "v2_focused_overlap",
        "v2_flags",
    ):
        assert {
            "research_notice",
            "recommendation_notice",
            "scope_notice",
        }.issubset(pd.read_csv(paths[key]).columns)


def test_v2_diagnostics_report_rejects_weight_calibration_when_gate_fails(
    tmp_path: Path,
) -> None:
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    _weak_v2_sample().to_csv(evaluated_path, index=False)
    paths = write_scoring_v2_diagnostics_outputs(
        evaluated_path,
        tmp_path / "weak",
        StrategyConfig(focused_count=1, candidate_count=2),
    )
    report = paths["v2_report"].read_text(encoding="utf-8")
    assert "当前不支持进入权重校准" in report
    assert "历史研究用途" in report


def test_v2_gate_passes_only_when_all_required_evidence_is_present() -> None:
    v1, v2 = _diagnostics_stub([1, 3, 5, 10, 20], improved_reversals=True)
    report = build_v2_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}),
        _factor_diagnostics_stub(),
        v1,
        v2,
        "stub.csv",
    )
    assert "当前支持进入权重校准" in report
    assert "2026-07" in report


def test_v2_gate_rejects_when_required_20d_evidence_is_missing() -> None:
    v1, v2 = _diagnostics_stub([1, 3, 5, 10, 30], improved_reversals=True)
    report = build_v2_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}),
        _factor_diagnostics_stub(),
        v1,
        v2,
        "stub.csv",
    )
    assert "当前不支持进入权重校准" in report
    assert "必须包含 20d 可比较 horizon：False" in report



def test_v2_gate_rejects_when_required_5d_10d_horizons_are_missing() -> None:
    v1, v2 = _diagnostics_stub([1, 2, 3, 4, 20], improved_reversals=True)
    report = build_v2_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}),
        _factor_diagnostics_stub(),
        v1,
        v2,
        "stub.csv",
    )
    assert "当前不支持进入权重校准" in report
    assert "gate 要求完整 [1, 3, 5, 10, 20]" in report


def test_v2_gate_rejects_when_quantile_dates_do_not_overlap() -> None:
    v1, v2 = _diagnostics_stub([1, 3, 5, 10, 20], improved_reversals=True)
    v2["quantile_monotonicity"] = v2["quantile_monotonicity"].assign(date="2026-08-01")
    report = build_v2_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}),
        _factor_diagnostics_stub(),
        v1,
        v2,
        "stub.csv",
    )
    assert "当前不支持进入权重校准" in report
    assert "分位反转共同可比较 horizon 数：0" in report

def test_v2_gate_rejects_when_quantile_comparison_is_missing() -> None:
    v1, v2 = _diagnostics_stub([1, 3, 5, 10, 20], improved_reversals=True)
    v2["quantile_monotonicity"] = pd.DataFrame(
        columns=["date", "horizon", "score_type", "q1_avg_return", "q4_avg_return", "q4_below_q1", "q4_below_q3"]
    )
    report = build_v2_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}),
        _factor_diagnostics_stub(),
        v1,
        v2,
        "stub.csv",
    )
    assert "当前不支持进入权重校准" in report
    assert "分位反转共同可比较 horizon 数：0" in report



def test_v2_gate_rejects_when_worst_date_comparison_is_missing() -> None:
    v1, v2 = _diagnostics_stub([1, 3, 5, 10, 20], improved_reversals=True)
    v2["by_date"] = pd.DataFrame(
        columns=["date", "horizon", "experimental_focused_minus_candidate_avg_return"]
    )
    report = build_v2_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}),
        _factor_diagnostics_stub(),
        v1,
        v2,
        "stub.csv",
    )
    assert "当前不支持进入权重校准" in report
    assert "最差日期共同可比较 horizon 数：0" in report

def test_scoring_v2_diagnostics_cli_runs(tmp_path: Path) -> None:
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    output_dir = tmp_path / "cli-v2"
    _v2_diagnostic_sample().to_csv(evaluated_path, index=False)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/scoring_v2_diagnostics.py",
            "--evaluated",
            str(evaluated_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "v2 diagnostics: HISTORICAL RESEARCH ONLY / NOT A RECOMMENDATION" in completed.stdout
    assert (output_dir / "2026-07-30_v2_diagnostics_report.md").is_file()
