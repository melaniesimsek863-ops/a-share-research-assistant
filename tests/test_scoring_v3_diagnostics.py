from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from a_share_ai.diagnostics.robustness import (
    EXPERIMENTAL_V2_PROFILE,
    EXPERIMENTAL_V3_PROFILE,
    build_scoring_robustness_diagnostics_for_profile,
)
from a_share_ai.diagnostics.scoring_v3 import (
    _build_v2_v3_focused_overlap,
    _gate_result,
    _with_csv_safety_notices,
    build_v3_diagnostics_report,
    write_scoring_v3_diagnostics_outputs,
)
from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import (
    score_experimental_v2_layer,
    score_experimental_v3_layer,
)


def _v3_diagnostic_sample() -> pd.DataFrame:
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
                    "signal_labels": "放量且收盘在20日均线上方,收盘在20日均线上方且60日回撤大于-8%,20日涨幅低于18%"
                    if index == 1
                    else "",
                    "return_20d": return_20d,
                    "return_60d": return_60d,
                    "drawdown_60d": -0.07 if index < 3 else -0.18,
                    "volume_ratio_5_20": 1.5 if index < 3 else 0.8,
                    "above_ma20": index < 3,
                    "ma_bullish": index < 3,
                    "risk_penalty": 0.0,
                    "forward_return_1d": forward_return,
                    "forward_return_3d": forward_return * 1.5,
                    "forward_return_5d": forward_return * 2.0,
                    "forward_return_10d": forward_return * 2.5,
                    "forward_return_20d": forward_return * 3.0,
                }
            )
    return pd.DataFrame(rows)


def _diagnostics_stub(
    horizons: list[int], *, v2_v3_focused_overlap_rate: float = 0.75
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    v2_horizon_rows, v3_horizon_rows, v2_date_rows, v3_date_rows = [], [], [], []
    v2_quantile_rows, v3_quantile_rows = [], []
    for horizon in horizons:
        v2_horizon_rows.append(
            {
                "horizon": horizon,
                "experimental_overall_delta_avg": 0.01,
                "experimental_worst_delta_avg": -0.03,
            }
        )
        v3_horizon_rows.append(
            {
                "horizon": horizon,
                "experimental_overall_delta_avg": 0.02
                if horizon in {1, 3, 5}
                else 0.011,
                "experimental_worst_delta_avg": -0.025,
            }
        )
        v2_date_rows.append(
            {
                "date": f"2026-07-{20 + horizon % 8:02d}",
                "horizon": horizon,
                "experimental_focused_minus_candidate_avg_return": -0.03,
            }
        )
        v3_date_rows.append(
            {
                "date": f"2026-07-{21 + horizon % 8:02d}",
                "horizon": horizon,
                "experimental_focused_minus_candidate_avg_return": -0.025,
            }
        )
        v2_quantile_rows.append(
            {
                "date": "2026-07-29",
                "horizon": horizon,
                "score_type": "experimental_v2_total_score",
                "q1_avg_return": 0.03,
                "q4_avg_return": 0.01,
                "q4_below_q1": True,
                "q4_below_q3": False,
            }
        )
        v3_quantile_rows.append(
            {
                "date": "2026-07-29",
                "horizon": horizon,
                "score_type": "experimental_v3_total_score",
                "q1_avg_return": 0.01,
                "q4_avg_return": 0.03,
                "q4_below_q1": False,
                "q4_below_q3": False,
            }
        )
    return (
        {
            "by_horizon": pd.DataFrame(v2_horizon_rows),
            "by_date": pd.DataFrame(v2_date_rows),
            "quantile_monotonicity": pd.DataFrame(v2_quantile_rows),
            "focused_overlap": pd.DataFrame(
                {"date": ["2026-07-29"], "focused_overlap_rate": [0.0]}
            ),
        },
        {
            "by_horizon": pd.DataFrame(v3_horizon_rows),
            "by_date": pd.DataFrame(v3_date_rows),
            "quantile_monotonicity": pd.DataFrame(v3_quantile_rows),
            "focused_overlap": pd.DataFrame(
                {"date": ["2026-07-29"], "focused_overlap_rate": [0.0]}
            ),
            "v2_v3_focused_overlap": pd.DataFrame(
                {
                    "date": ["2026-07-29"],
                    "v2_v3_focused_overlap_rate": [v2_v3_focused_overlap_rate],
                }
            ),
        },
    )


def _report(v2: dict[str, pd.DataFrame], v3: dict[str, pd.DataFrame]) -> str:
    return build_v3_diagnostics_report(
        pd.DataFrame({"code": ["000001"]}), v2, v3, "stub.csv"
    )


def test_v3_profile_requires_v3_columns_and_adds_profile_name() -> None:
    with pytest.raises(ValueError, match="experimental_v3_total_score"):
        build_scoring_robustness_diagnostics_for_profile(
            _v3_diagnostic_sample(),
            StrategyConfig(focused_count=1, candidate_count=2),
            EXPERIMENTAL_V3_PROFILE,
        )
    scored = score_experimental_v3_layer(
        _v3_diagnostic_sample(), StrategyConfig(focused_count=1, candidate_count=2)
    )
    diagnostics = build_scoring_robustness_diagnostics_for_profile(
        scored,
        StrategyConfig(focused_count=1, candidate_count=2),
        EXPERIMENTAL_V3_PROFILE,
    )
    assert set(diagnostics["by_horizon"]["score_profile"]) == {"experimental_v3"}


def test_write_scoring_v3_diagnostics_outputs_creates_artifacts_and_safety_notices(
    tmp_path: Path,
) -> None:
    evaluated_path = tmp_path / "2026-07-30_replay_evaluated.csv"
    _v3_diagnostic_sample().to_csv(evaluated_path, index=False)
    output_dir = tmp_path / "v3"
    paths = write_scoring_v3_diagnostics_outputs(
        evaluated_path, output_dir, StrategyConfig(focused_count=1, candidate_count=2)
    )
    assert paths == {
        "factor_report": output_dir / "2026-07-30_factor_effectiveness_report.md",
        "factor_by_factor": output_dir
        / "2026-07-30_factor_effectiveness_by_factor.csv",
        "factor_by_date": output_dir / "2026-07-30_factor_effectiveness_by_date.csv",
        "v3_report": output_dir / "2026-07-30_v3_diagnostics_report.md",
        "v3_summary": output_dir / "2026-07-30_v3_robustness_summary.md",
        "v3_by_date": output_dir / "2026-07-30_v3_robustness_by_date.csv",
        "v3_by_horizon": output_dir / "2026-07-30_v3_robustness_by_horizon.csv",
        "v3_quantile_monotonicity": output_dir
        / "2026-07-30_v3_robustness_quantile_monotonicity.csv",
        "v3_focused_overlap": output_dir
        / "2026-07-30_v3_robustness_focused_overlap.csv",
        "v3_flags": output_dir / "2026-07-30_v3_robustness_flags.csv",
    }
    report = paths["v3_report"].read_text(encoding="utf-8")
    assert "V3 SCORING DIAGNOSTICS / 实验评分 v3 诊断" in report
    assert "v2 vs v3" in report
    assert "v2/v3 focused 股票池平均重合率：" in report
    assert "- v3 focused 股票池平均重合率：" not in report
    assert "不构成投资建议" in report
    for key in ("factor_report", "v3_report", "v3_summary"):
        markdown = paths[key].read_text(encoding="utf-8")
        assert "HISTORICAL RESEARCH ONLY / 历史研究用途" in markdown
        assert "NOT A RECOMMENDATION / 不构成投资建议" in markdown
    for key in (
        "factor_by_factor",
        "factor_by_date",
        "v3_by_date",
        "v3_by_horizon",
        "v3_quantile_monotonicity",
        "v3_focused_overlap",
        "v3_flags",
    ):
        artifact = pd.read_csv(paths[key])
        assert {"research_notice", "recommendation_notice", "scope_notice"}.issubset(
            artifact.columns
        )
        assert (
            artifact["research_notice"]
            .eq("HISTORICAL RESEARCH ONLY / 历史研究用途")
            .all()
        )
        assert (
            artifact["recommendation_notice"]
            .eq("NOT A RECOMMENDATION / 不构成投资建议")
            .all()
        )


def test_v3_gate_passes_with_complete_required_evidence() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    report = _report(v2, v3)
    assert "当前支持进入权重校准" in report


def test_v2_v3_focused_overlap_uses_experimental_tiers_not_production_tier() -> None:
    v2_scored = pd.DataFrame(
        {
            "date": ["2026-07-29"] * 3,
            "code": ["000001", "000002", "000003"],
            "tier": ["focused", "candidate", "candidate"],
            "experimental_v2_tier": ["candidate", "focused", "candidate"],
        }
    )
    v3_scored = v2_scored.assign(
        experimental_v3_tier=["candidate", "focused", "candidate"]
    )

    overlap = _build_v2_v3_focused_overlap(v2_scored, v3_scored)

    row = overlap.iloc[0]
    assert row["experimental_v2_focused_count"] == 1
    assert row["experimental_v3_focused_count"] == 1
    assert row["v2_v3_focused_overlap_count"] == 1
    assert row["v2_v3_focused_overlap_rate"] == 1.0

def test_v3_gate_rejects_missing_20d_evidence() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 30])
    report = _report(v2, v3)
    assert "当前不支持进入权重校准" in report
    assert "必须包含 20d 可比较 horizon：False" in report


def test_v3_gate_rejects_incomplete_required_horizons() -> None:
    v2, v3 = _diagnostics_stub([1, 2, 3, 4, 20])
    report = _report(v2, v3)
    assert "当前不支持进入权重校准" in report
    assert "gate 要求完整 [1, 3, 5, 10, 20]" in report


def test_v3_gate_rejects_quantile_comparison_gaps() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    v3["quantile_monotonicity"] = v3["quantile_monotonicity"].assign(date="2026-08-01")
    report = _report(v2, v3)
    assert "当前不支持进入权重校准" in report
    assert "分位反转共同可比较 horizon 数：0" in report


def test_v3_gate_rejects_worst_date_comparison_gaps() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    v3["by_date"] = pd.DataFrame(
        columns=["date", "horizon", "experimental_focused_minus_candidate_avg_return"]
    )
    report = _report(v2, v3)
    assert "当前不支持进入权重校准" in report
    assert "最差日期共同可比较 horizon 数：0" in report


def test_v3_gate_allows_worst_date_degradation_within_material_threshold() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    v3["by_horizon"].loc[
        v3["by_horizon"]["horizon"].eq(20), "experimental_worst_delta_avg"
    ] = -0.0301
    report = _report(v2, v3)
    assert "当前支持进入权重校准" in report
    assert "最差日期风险未变差：True" in report


def test_csv_safety_helper_fills_missing_or_incorrect_notice_values() -> None:
    artifact = _with_csv_safety_notices(
        pd.DataFrame(
            {
                "research_notice": ["incorrect"],
                "recommendation_notice": [None],
                "scope_notice": [""],
            }
        )
    )
    assert artifact["research_notice"].eq("HISTORICAL RESEARCH ONLY / 历史研究用途").all()
    assert artifact["recommendation_notice"].eq("NOT A RECOMMENDATION / 不构成投资建议").all()
    assert artifact["scope_notice"].eq("LIMITED CACHED REPLAY / 有限缓存池回放").all()
def test_v3_gate_rejects_low_average_focused_overlap() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20], v2_v3_focused_overlap_rate=0.49)
    report = _report(v2, v3)
    assert "当前不支持进入权重校准" in report
    assert "focused 股票池平均重合率不低于 50%：False" in report


def test_v3_report_lists_dates_with_focused_overlap_below_20_percent() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    v3["v2_v3_focused_overlap"] = pd.DataFrame(
        {
            "date": ["2026-07-29", "2026-07-30"],
            "v2_v3_focused_overlap_rate": [0.19, 0.20],
        }
    )
    report = _report(v2, v3)
    assert "2026-07-29 (19.00%)" in report
    assert "2026-07-30 (20.00%)" not in report


def test_v3_gate_rejects_worst_date_degradation_beyond_material_threshold() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    v3["by_horizon"].loc[
        v3["by_horizon"]["horizon"].eq(20), "experimental_worst_delta_avg"
    ] = -0.0321
    assert _gate_result(v2, v3)[0] is False


def test_v3_gate_rejects_all_negative_horizons_when_other_gates_pass() -> None:
    v2, v3 = _diagnostics_stub([1, 3, 5, 10, 20])
    v2["by_horizon"]["experimental_overall_delta_avg"] = -0.02
    v3["by_horizon"]["experimental_overall_delta_avg"] = -0.01

    gate_pass, _ = _gate_result(v2, v3)
    assert gate_pass is False


def test_v3_profile_preserves_v2_baseline_diagnostics() -> None:
    v2_scored = score_experimental_v2_layer(
        _v3_diagnostic_sample(), StrategyConfig(focused_count=1, candidate_count=2)
    )
    diagnostics = build_scoring_robustness_diagnostics_for_profile(
        v2_scored,
        StrategyConfig(focused_count=1, candidate_count=2),
        EXPERIMENTAL_V2_PROFILE,
    )
    assert set(diagnostics["by_horizon"]["score_profile"]) == {"experimental_v2"}
