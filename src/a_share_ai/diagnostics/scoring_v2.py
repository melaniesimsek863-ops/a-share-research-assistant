from __future__ import annotations

from pathlib import Path

import pandas as pd

from a_share_ai.diagnostics.factor_effectiveness import (
    build_factor_effectiveness_diagnostics,
    build_factor_effectiveness_report,
)
from a_share_ai.diagnostics.robustness import (
    EXPERIMENTAL_V2_PROFILE,
    RECOMMENDATION_NOTICE,
    RESEARCH_NOTICE,
    SCOPE_NOTICE,
    build_scoring_robustness_diagnostics,
    build_scoring_robustness_diagnostics_for_profile,
    build_scoring_robustness_report,
)
from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import (
    score_experimental_layer,
    score_experimental_v2_layer,
)

MATERIAL_WORSE_THRESHOLD = 0.002
REQUIRED_GATE_HORIZONS = {1, 3, 5, 10, 20}
REQUIRED_GATE_HORIZON_COUNT = 5
REQUIRED_GATE_HORIZON = 20


def _format_percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(numeric) else f"{numeric:.2%}"


def _horizon_map(by_horizon: pd.DataFrame, column: str) -> dict[int, float]:
    if by_horizon.empty:
        return {}
    return {
        int(row["horizon"]): float(row[column])
        for _, row in by_horizon.iterrows()
        if pd.notna(row[column])
    }


def _worst_date_map(by_date: pd.DataFrame) -> dict[int, tuple[str, float]]:
    if by_date.empty:
        return {}
    result: dict[int, tuple[str, float]] = {}
    for horizon, group in by_date.groupby("horizon", dropna=False):
        comparable = group.loc[group["experimental_focused_minus_candidate_avg_return"].notna()]
        if comparable.empty:
            continue
        worst = comparable.loc[comparable["experimental_focused_minus_candidate_avg_return"].idxmin()]
        result[int(horizon)] = (
            str(worst["date"]),
            float(worst["experimental_focused_minus_candidate_avg_return"]),
        )
    return result


def _quantile_evidence(quantile: pd.DataFrame, score_type: str) -> pd.DataFrame:
    columns = ["date", "horizon", "is_reversal"]
    if quantile.empty:
        return pd.DataFrame(columns=columns)
    subset = quantile.loc[quantile["score_type"].eq(score_type)].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    usable = subset.loc[subset[["q1_avg_return", "q4_avg_return"]].notna().all(axis=1)].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)
    usable["date"] = usable["date"].astype(str)
    usable["horizon"] = usable["horizon"].astype(int)
    usable["is_reversal"] = usable["q4_below_q1"].astype(bool) | usable["q4_below_q3"].astype(bool)
    return usable[columns]


def _reversal_counts_by_horizon(quantile: pd.DataFrame, score_type: str) -> dict[int, int]:
    evidence = _quantile_evidence(quantile, score_type)
    if evidence.empty:
        return {}
    reversal = evidence.loc[evidence["is_reversal"]]
    return {int(horizon): len(group) for horizon, group in reversal.groupby("horizon")}


def _common_quantile_evidence(v1_quantile: pd.DataFrame, v2_quantile: pd.DataFrame) -> pd.DataFrame:
    v1_evidence = _quantile_evidence(v1_quantile, "experimental_total_score").rename(
        columns={"is_reversal": "v1_reversal"}
    )
    v2_evidence = _quantile_evidence(v2_quantile, "experimental_v2_total_score").rename(
        columns={"is_reversal": "v2_reversal"}
    )
    if v1_evidence.empty or v2_evidence.empty:
        return pd.DataFrame(columns=["date", "horizon", "v1_reversal", "v2_reversal"])
    return v1_evidence.merge(
        v2_evidence,
        on=["date", "horizon"],
        how="inner",
        validate="one_to_one",
    )

def _top_bottom_factor_lines(factor_diagnostics: dict[str, pd.DataFrame]) -> list[str]:
    by_factor = factor_diagnostics.get("by_factor", pd.DataFrame())
    lines = ["## 因子方向摘要", ""]
    if by_factor.empty:
        return lines + ["- 没有可用因子诊断。"]
    ranked = by_factor.sort_values("q4_minus_q1_mean", ascending=False)
    lines.append("- Top factors（按 q4-q1 均值，最多 5 条）：")
    for row in ranked.head(5).itertuples():
        lines.append(
            f"  - {row.factor}/{int(row.horizon)}d: {row.direction}, "
            f"q4-q1={_format_percent(row.q4_minus_q1_mean)}, "
            f"corr={float(row.mean_spearman_corr):.3f}"
        )
    lines.append("- Bottom / negative factors（最多 5 条）：")
    for row in ranked.tail(5).sort_values("q4_minus_q1_mean").itertuples():
        lines.append(
            f"  - {row.factor}/{int(row.horizon)}d: {row.direction}, "
            f"q4-q1={_format_percent(row.q4_minus_q1_mean)}, "
            f"corr={float(row.mean_spearman_corr):.3f}"
        )
    return lines


def _gate_result(v1: dict[str, pd.DataFrame], v2: dict[str, pd.DataFrame]) -> tuple[bool, list[str]]:
    v1_delta = _horizon_map(v1["by_horizon"], "experimental_overall_delta_avg")
    v2_delta = _horizon_map(v2["by_horizon"], "experimental_overall_delta_avg")
    v1_worst = _horizon_map(v1["by_horizon"], "experimental_worst_delta_avg")
    v2_worst = _horizon_map(v2["by_horizon"], "experimental_worst_delta_avg")
    v1_worst_dates = _worst_date_map(v1["by_date"])
    v2_worst_dates = _worst_date_map(v2["by_date"])
    common_horizons = sorted(set(v1_delta) & set(v2_delta))
    required_horizons_present = REQUIRED_GATE_HORIZONS.issubset(common_horizons)

    coverage_ok = required_horizons_present
    horizon_20_present = REQUIRED_GATE_HORIZON in common_horizons
    non_negative_count = sum(1 for horizon in REQUIRED_GATE_HORIZONS if v2_delta.get(horizon, float("nan")) >= 0)

    common_quantile = _common_quantile_evidence(v1["quantile_monotonicity"], v2["quantile_monotonicity"])
    quantile_comparable_horizons = set(common_quantile["horizon"].unique()) if not common_quantile.empty else set()
    quantile_coverage_ok = REQUIRED_GATE_HORIZONS.issubset(quantile_comparable_horizons)
    reduced_reversal_count = 0
    if not common_quantile.empty:
        for horizon in REQUIRED_GATE_HORIZONS:
            horizon_rows = common_quantile.loc[common_quantile["horizon"].eq(horizon)]
            if not horizon_rows.empty and int(horizon_rows["v2_reversal"].sum()) < int(horizon_rows["v1_reversal"].sum()):
                reduced_reversal_count += 1

    horizon_20_ok = (
        horizon_20_present
        and v2_delta[REQUIRED_GATE_HORIZON]
        >= v1_delta[REQUIRED_GATE_HORIZON] - MATERIAL_WORSE_THRESHOLD
    )
    worst_comparable_horizons = set(common_horizons) & set(v1_worst) & set(v2_worst) & set(v1_worst_dates) & set(v2_worst_dates)
    worst_coverage_ok = REQUIRED_GATE_HORIZONS.issubset(worst_comparable_horizons)
    worst_date_ok = worst_coverage_ok and all(
        v2_worst[horizon] >= v1_worst[horizon] - MATERIAL_WORSE_THRESHOLD
        for horizon in REQUIRED_GATE_HORIZONS
    )

    gate_pass = (
        coverage_ok
        and horizon_20_ok
        and quantile_coverage_ok
        and worst_coverage_ok
        and non_negative_count >= 3
        and reduced_reversal_count >= 3
        and worst_date_ok
    )
    reasons = [
        f"v1/v2 共同可比较 horizon 数：{len(common_horizons)}（gate 要求完整 {sorted(REQUIRED_GATE_HORIZONS)}）。",
        f"必须包含 20d 可比较 horizon：{horizon_20_present}。",
        f"v2 focused-candidate 平均差非负 horizon：{non_negative_count}/5。",
        f"分位反转共同可比较 horizon 数：{len(quantile_comparable_horizons)}；v2 反转少于 v1 的 horizon：{reduced_reversal_count}/5。",
        f"20d 未明显变差：{horizon_20_ok}。",
        f"最差日期共同可比较 horizon 数：{len(worst_comparable_horizons)}；最差日期风险未变差：{worst_date_ok}。",
    ]
    return gate_pass, reasons

def _comparison_table(v1: dict[str, pd.DataFrame], v2: dict[str, pd.DataFrame]) -> list[str]:
    v1_delta = _horizon_map(v1["by_horizon"], "experimental_overall_delta_avg")
    v2_delta = _horizon_map(v2["by_horizon"], "experimental_overall_delta_avg")
    v1_worst = _worst_date_map(v1["by_date"])
    v2_worst = _worst_date_map(v2["by_date"])
    v1_reversals = _reversal_counts_by_horizon(v1["quantile_monotonicity"], "experimental_total_score")
    v2_reversals = _reversal_counts_by_horizon(v2["quantile_monotonicity"], "experimental_v2_total_score")
    horizons = sorted(set(v1_delta) | set(v2_delta))
    lines = [
        "## v1 vs v2 horizon comparison",
        "",
        "| Horizon | v1 focused-candidate avg | v2 focused-candidate avg | v1 reversals | v2 reversals | v1 worst date | v2 worst date |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for horizon in horizons:
        v1_worst_text = "n/a"
        v2_worst_text = "n/a"
        if horizon in v1_worst:
            v1_worst_text = f"{v1_worst[horizon][0]} ({_format_percent(v1_worst[horizon][1])})"
        if horizon in v2_worst:
            v2_worst_text = f"{v2_worst[horizon][0]} ({_format_percent(v2_worst[horizon][1])})"
        lines.append(
            f"| {horizon}d | {_format_percent(v1_delta.get(horizon))} | "
            f"{_format_percent(v2_delta.get(horizon))} | "
            f"{v1_reversals.get(horizon, 0)} | {v2_reversals.get(horizon, 0)} | "
            f"{v1_worst_text} | {v2_worst_text} |"
        )
    return lines


def build_v2_diagnostics_report(
    evaluated: pd.DataFrame,
    factor_diagnostics: dict[str, pd.DataFrame],
    v1_diagnostics: dict[str, pd.DataFrame],
    v2_diagnostics: dict[str, pd.DataFrame],
    source_name: str,
) -> str:
    gate_pass, gate_reasons = _gate_result(v1_diagnostics, v2_diagnostics)
    lines = [
        "# V2 SCORING DIAGNOSTICS / 实验评分 v2 诊断",
        "",
        RESEARCH_NOTICE,
        RECOMMENDATION_NOTICE,
        SCOPE_NOTICE,
        "",
        f"- source: {source_name}",
        f"- input rows: {len(evaluated)}",
        "- scope: historical cached replay only; production scoring, reports, Excel outputs and execution flows remain unchanged.",
        "",
    ]
    lines.extend(_top_bottom_factor_lines(factor_diagnostics))
    lines.extend([""])
    lines.extend(_comparison_table(v1_diagnostics, v2_diagnostics))
    lines.extend(["", "## 下一阶段判断", ""])
    lines.extend(f"- {reason}" for reason in gate_reasons)
    if gate_pass:
        lines.append("- 当前支持进入权重校准，但仅限历史研究流程；仍不构成投资建议。")
    else:
        lines.append("- 当前不支持进入权重校准，建议返回因子设计、评分链路或数据覆盖诊断。")
    lines.extend(
        [
            "",
            "## 样本限制",
            "",
            "- LIMITED CACHED REPLAY / 有限缓存池回放：该结论只来自本地 cached replay 文件。",
            "- NOT A RECOMMENDATION / 不构成投资建议：不得据此直接买卖、加仓或减仓。",
        ]
    )
    return "\n".join(lines) + "\n"


def _gate_sync_section(v1: dict[str, pd.DataFrame], v2: dict[str, pd.DataFrame]) -> str:
    gate_pass, reasons = _gate_result(v1, v2)
    lines = ["", "## v2 gate 同步结论", ""]
    lines.extend(f"- {reason}" for reason in reasons)
    if gate_pass:
        lines.append("- 当前支持进入权重校准，但仅限历史研究流程；仍不构成投资建议。")
    else:
        lines.append("- 当前不支持进入权重校准，建议返回因子设计、评分链路或数据覆盖诊断。")
    return "\n".join(lines) + "\n"


def _report_date_from(evaluated: pd.DataFrame, source_path: Path) -> str:
    if "date" in evaluated.columns and not evaluated.empty:
        parsed = pd.to_datetime(evaluated["date"], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().date().isoformat()
    return source_path.stem.replace("_replay_evaluated", "")


def write_scoring_v2_diagnostics_outputs(
    evaluated_path: str | Path,
    output_dir: str | Path,
    config: StrategyConfig,
) -> dict[str, Path]:
    source_path = Path(evaluated_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluated = pd.read_csv(source_path, dtype={"code": str})
    report_date = _report_date_from(evaluated, source_path)

    v1_scored = score_experimental_layer(evaluated, config)
    factor_diagnostics = build_factor_effectiveness_diagnostics(v1_scored)
    v1_diagnostics = build_scoring_robustness_diagnostics(v1_scored, config)
    v2_scored = score_experimental_v2_layer(v1_scored, config)
    v2_diagnostics = build_scoring_robustness_diagnostics_for_profile(
        v2_scored,
        config,
        EXPERIMENTAL_V2_PROFILE,
    )

    paths = {
        "factor_report": output_path / f"{report_date}_factor_effectiveness_report.md",
        "factor_by_factor": output_path / f"{report_date}_factor_effectiveness_by_factor.csv",
        "factor_by_date": output_path / f"{report_date}_factor_effectiveness_by_date.csv",
        "v2_report": output_path / f"{report_date}_v2_diagnostics_report.md",
        "v2_summary": output_path / f"{report_date}_v2_robustness_summary.md",
        "v2_by_date": output_path / f"{report_date}_v2_robustness_by_date.csv",
        "v2_by_horizon": output_path / f"{report_date}_v2_robustness_by_horizon.csv",
        "v2_quantile_monotonicity": output_path / f"{report_date}_v2_robustness_quantile_monotonicity.csv",
        "v2_focused_overlap": output_path / f"{report_date}_v2_robustness_focused_overlap.csv",
        "v2_flags": output_path / f"{report_date}_v2_robustness_flags.csv",
    }
    paths["factor_report"].write_text(
        build_factor_effectiveness_report(factor_diagnostics, source_path.name),
        encoding="utf-8",
    )
    factor_diagnostics["by_factor"].to_csv(paths["factor_by_factor"], index=False)
    factor_diagnostics["by_date"].to_csv(paths["factor_by_date"], index=False)
    combined_report = build_v2_diagnostics_report(
        evaluated,
        factor_diagnostics,
        v1_diagnostics,
        v2_diagnostics,
        source_path.name,
    )
    paths["v2_report"].write_text(combined_report, encoding="utf-8")
    v2_summary = build_scoring_robustness_report(v2_diagnostics, source_path.name)
    v2_summary += _gate_sync_section(v1_diagnostics, v2_diagnostics)
    paths["v2_summary"].write_text(v2_summary, encoding="utf-8")
    for key, diagnostic_key in (
        ("v2_by_date", "by_date"),
        ("v2_by_horizon", "by_horizon"),
        ("v2_quantile_monotonicity", "quantile_monotonicity"),
        ("v2_focused_overlap", "focused_overlap"),
        ("v2_flags", "flags"),
    ):
        v2_diagnostics[diagnostic_key].to_csv(paths[key], index=False)
    return paths
