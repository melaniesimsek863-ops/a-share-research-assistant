from __future__ import annotations

from pathlib import Path

import pandas as pd

from a_share_ai.diagnostics.factor_effectiveness import (
    build_factor_effectiveness_diagnostics,
    build_factor_effectiveness_report,
)
from a_share_ai.diagnostics.robustness import (
    EXPERIMENTAL_V2_PROFILE,
    EXPERIMENTAL_V3_PROFILE,
    RECOMMENDATION_NOTICE,
    RESEARCH_NOTICE,
    SCOPE_NOTICE,
    build_scoring_robustness_diagnostics_for_profile,
    build_scoring_robustness_report,
)
from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import (
    score_experimental_layer,
    score_experimental_v2_layer,
    score_experimental_v3_layer,
)

MATERIAL_WORSE_THRESHOLD = 0.002
REQUIRED_GATE_HORIZONS = {1, 3, 5, 10, 20}
REQUIRED_GATE_HORIZON = 20
MIN_AVERAGE_FOCUSED_OVERLAP = 0.50
LOW_FOCUSED_OVERLAP_THRESHOLD = 0.20

CSV_SAFETY_NOTICES = {
    "research_notice": RESEARCH_NOTICE,
    "recommendation_notice": RECOMMENDATION_NOTICE,
    "scope_notice": SCOPE_NOTICE,
}


def _with_csv_safety_notices(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column, notice in CSV_SAFETY_NOTICES.items():
        result[column] = notice
    return result

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
        if not comparable.empty:
            worst = comparable.loc[
                comparable["experimental_focused_minus_candidate_avg_return"].idxmin()
            ]
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
    usable = subset.loc[subset[["q1_avg_return", "q4_avg_return"]].notna().all(axis=1)].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)
    usable["date"] = usable["date"].astype(str)
    usable["horizon"] = usable["horizon"].astype(int)
    usable["is_reversal"] = usable["q4_below_q1"].astype(bool) | usable["q4_below_q3"].astype(bool)
    return usable[columns]


def _common_quantile_evidence(v2_quantile: pd.DataFrame, v3_quantile: pd.DataFrame) -> pd.DataFrame:
    v2_evidence = _quantile_evidence(v2_quantile, "experimental_v2_total_score").rename(
        columns={"is_reversal": "v2_reversal"}
    )
    v3_evidence = _quantile_evidence(v3_quantile, "experimental_v3_total_score").rename(
        columns={"is_reversal": "v3_reversal"}
    )
    if v2_evidence.empty or v3_evidence.empty:
        return pd.DataFrame(columns=["date", "horizon", "v2_reversal", "v3_reversal"])
    return v2_evidence.merge(v3_evidence, on=["date", "horizon"], how="inner", validate="one_to_one")


def _reversal_counts_by_horizon(quantile: pd.DataFrame, score_type: str) -> dict[int, int]:
    evidence = _quantile_evidence(quantile, score_type)
    if evidence.empty:
        return {}
    return {int(horizon): int(group["is_reversal"].sum()) for horizon, group in evidence.groupby("horizon")}


def _focused_overlap_summary(focused_overlap: pd.DataFrame) -> tuple[float, list[str]]:
    if focused_overlap.empty or "focused_overlap_rate" not in focused_overlap:
        return float("nan"), []
    rates = pd.to_numeric(focused_overlap["focused_overlap_rate"], errors="coerce")
    average = float(rates.mean()) if rates.notna().any() else float("nan")
    low_dates = [
        f"{row.date} ({_format_percent(row.focused_overlap_rate)})"
        for row in focused_overlap.loc[rates < LOW_FOCUSED_OVERLAP_THRESHOLD, ["date", "focused_overlap_rate"]].itertuples(index=False)
    ]
    return average, low_dates

def _build_v2_v3_focused_overlap(v2_scored: pd.DataFrame, v3_scored: pd.DataFrame) -> pd.DataFrame:
    v2_focused_by_date = {
        str(date_value): set(
            group.loc[group["experimental_v2_tier"].eq("focused"), "code"].astype(str)
        )
        for date_value, group in v2_scored.groupby("date", dropna=False)
    }
    v3_focused_by_date = {
        str(date_value): set(
            group.loc[group["experimental_v3_tier"].eq("focused"), "code"].astype(str)
        )
        for date_value, group in v3_scored.groupby("date", dropna=False)
    }
    rows = []
    for date_value in sorted(set(v2_focused_by_date) | set(v3_focused_by_date)):
        v2_codes = v2_focused_by_date.get(date_value, set())
        v3_codes = v3_focused_by_date.get(date_value, set())
        denominator = len(v2_codes | v3_codes)
        rows.append(
            {
                "date": date_value,
                "experimental_v2_focused_count": len(v2_codes),
                "experimental_v3_focused_count": len(v3_codes),
                "v2_v3_focused_overlap_count": len(v2_codes & v3_codes),
                "v2_v3_focused_overlap_rate": len(v2_codes & v3_codes) / denominator
                if denominator
                else 0.0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "experimental_v2_focused_count",
            "experimental_v3_focused_count",
            "v2_v3_focused_overlap_count",
            "v2_v3_focused_overlap_rate",
        ],
    )


def _v2_v3_focused_overlap_summary(focused_overlap: pd.DataFrame) -> tuple[float, list[str]]:
    if focused_overlap.empty or "v2_v3_focused_overlap_rate" not in focused_overlap:
        return float("nan"), []
    rates = pd.to_numeric(focused_overlap["v2_v3_focused_overlap_rate"], errors="coerce")
    average = float(rates.mean()) if rates.notna().any() else float("nan")
    low_dates = [
        f"{row.date} ({_format_percent(row.v2_v3_focused_overlap_rate)})"
        for row in focused_overlap.loc[rates < LOW_FOCUSED_OVERLAP_THRESHOLD, ["date", "v2_v3_focused_overlap_rate"]].itertuples(index=False)
    ]
    return average, low_dates


def _gate_result(v2: dict[str, pd.DataFrame], v3: dict[str, pd.DataFrame]) -> tuple[bool, list[str]]:
    v2_delta = _horizon_map(v2["by_horizon"], "experimental_overall_delta_avg")
    v3_delta = _horizon_map(v3["by_horizon"], "experimental_overall_delta_avg")
    v2_worst = _horizon_map(v2["by_horizon"], "experimental_worst_delta_avg")
    v3_worst = _horizon_map(v3["by_horizon"], "experimental_worst_delta_avg")
    v2_worst_dates = _worst_date_map(v2["by_date"])
    v3_worst_dates = _worst_date_map(v3["by_date"])
    common_horizons = sorted(set(v2_delta) & set(v3_delta))
    coverage_ok = REQUIRED_GATE_HORIZONS.issubset(common_horizons)
    horizon_20_present = REQUIRED_GATE_HORIZON in common_horizons
    non_negative_count = sum(v3_delta.get(horizon, float("nan")) >= 0 for horizon in REQUIRED_GATE_HORIZONS)

    common_quantile = _common_quantile_evidence(v2["quantile_monotonicity"], v3["quantile_monotonicity"])
    quantile_comparable_horizons = set(common_quantile["horizon"].unique()) if not common_quantile.empty else set()
    quantile_coverage_ok = REQUIRED_GATE_HORIZONS.issubset(quantile_comparable_horizons)
    reduced_reversal_count = sum(
        int(rows["v3_reversal"].sum()) < int(rows["v2_reversal"].sum())
        for horizon in REQUIRED_GATE_HORIZONS
        if not (rows := common_quantile.loc[common_quantile["horizon"].eq(horizon)]).empty
    )

    horizon_20_ok = horizon_20_present and (
        v3_delta[REQUIRED_GATE_HORIZON]
        >= v2_delta[REQUIRED_GATE_HORIZON] - MATERIAL_WORSE_THRESHOLD
    )
    worst_comparable_horizons = set(common_horizons) & set(v2_worst) & set(v3_worst) & set(v2_worst_dates) & set(v3_worst_dates)
    worst_coverage_ok = REQUIRED_GATE_HORIZONS.issubset(worst_comparable_horizons)
    worst_date_ok = worst_coverage_ok and all(
        v3_worst[horizon] >= v2_worst[horizon] - MATERIAL_WORSE_THRESHOLD
        for horizon in REQUIRED_GATE_HORIZONS
    )
    average_overlap, _ = _v2_v3_focused_overlap_summary(v3["v2_v3_focused_overlap"])
    focused_overlap_ok = pd.notna(average_overlap) and average_overlap >= MIN_AVERAGE_FOCUSED_OVERLAP

    gate_pass = (
        coverage_ok and horizon_20_ok and quantile_coverage_ok and worst_coverage_ok
        and non_negative_count >= 3 and reduced_reversal_count >= 3 and worst_date_ok
        and focused_overlap_ok
    )
    reasons = [
        f"v2/v3 共同可比较 horizon 数：{len(common_horizons)}（gate 要求完整 {sorted(REQUIRED_GATE_HORIZONS)}）。",
        f"必须包含 20d 可比较 horizon：{horizon_20_present}。",
        f"v3 focused-candidate 平均差非负 horizon：{non_negative_count}/5。",
        f"分位反转共同可比较 horizon 数：{len(quantile_comparable_horizons)}；v3 反转少于 v2 的 horizon：{reduced_reversal_count}/5。",
        f"20d 未明显变差：{horizon_20_ok}。",
        f"最差日期共同可比较 horizon 数：{len(worst_comparable_horizons)}；最差日期风险未变差：{worst_date_ok}。",
        f"focused 股票池平均重合率不低于 50%：{focused_overlap_ok}。",
    ]
    return gate_pass, reasons


def _comparison_table(v2: dict[str, pd.DataFrame], v3: dict[str, pd.DataFrame]) -> list[str]:
    v2_delta = _horizon_map(v2["by_horizon"], "experimental_overall_delta_avg")
    v3_delta = _horizon_map(v3["by_horizon"], "experimental_overall_delta_avg")
    v2_worst = _worst_date_map(v2["by_date"])
    v3_worst = _worst_date_map(v3["by_date"])
    v2_reversals = _reversal_counts_by_horizon(v2["quantile_monotonicity"], "experimental_v2_total_score")
    v3_reversals = _reversal_counts_by_horizon(v3["quantile_monotonicity"], "experimental_v3_total_score")
    lines = ["## v2 vs v3 horizon comparison", "", "| Horizon | v2 focused-candidate avg | v3 focused-candidate avg | v2 reversals | v3 reversals | v2 worst date | v3 worst date |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for horizon in sorted(set(v2_delta) | set(v3_delta)):
        v2_worst_text = f"{v2_worst[horizon][0]} ({_format_percent(v2_worst[horizon][1])})" if horizon in v2_worst else "n/a"
        v3_worst_text = f"{v3_worst[horizon][0]} ({_format_percent(v3_worst[horizon][1])})" if horizon in v3_worst else "n/a"
        lines.append(f"| {horizon}d | {_format_percent(v2_delta.get(horizon))} | {_format_percent(v3_delta.get(horizon))} | {v2_reversals.get(horizon, 0)} | {v3_reversals.get(horizon, 0)} | {v2_worst_text} | {v3_worst_text} |")
    return lines


def build_v3_diagnostics_report(evaluated: pd.DataFrame, v2_diagnostics: dict[str, pd.DataFrame], v3_diagnostics: dict[str, pd.DataFrame], source_name: str) -> str:
    gate_pass, gate_reasons = _gate_result(v2_diagnostics, v3_diagnostics)
    average_overlap, low_dates = _v2_v3_focused_overlap_summary(v3_diagnostics["v2_v3_focused_overlap"])
    lines = ["# V3 SCORING DIAGNOSTICS / 实验评分 v3 诊断", "", RESEARCH_NOTICE, RECOMMENDATION_NOTICE, SCOPE_NOTICE, "", f"- source: {source_name}", f"- input rows: {len(evaluated)}", "- scope: historical cached replay only; production scoring, reports, Excel outputs and execution flows remain unchanged.", ""]
    lines.extend(_comparison_table(v2_diagnostics, v3_diagnostics))
    lines.extend(["", "## focused 股票池重合", "", f"- v2/v3 focused 股票池平均重合率：{_format_percent(average_overlap)}。"])
    if low_dates:
        lines.append(f"- 低于 20% 的日期：{', '.join(low_dates)}。")
    lines.extend(["", "## 下一阶段判断", ""])
    lines.extend(f"- {reason}" for reason in gate_reasons)
    lines.append("- 当前支持进入权重校准，但仅限历史研究流程；仍不构成投资建议。" if gate_pass else "- 当前不支持进入权重校准，建议返回因子设计、评分链路或数据覆盖诊断。")
    lines.extend(["", "## 样本限制", "", "- LIMITED CACHED REPLAY / 有限缓存池回放：该结论只来自本地 cached replay 文件。", "- NOT A RECOMMENDATION / 不构成投资建议：不得据此直接买卖、加仓或减仓。"])
    return "\n".join(lines) + "\n"


def _gate_sync_section(v2: dict[str, pd.DataFrame], v3: dict[str, pd.DataFrame]) -> str:
    gate_pass, reasons = _gate_result(v2, v3)
    lines = ["", "## v3 gate 同步结论", ""]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("- 当前支持进入权重校准，但仅限历史研究流程；仍不构成投资建议。" if gate_pass else "- 当前不支持进入权重校准，建议返回因子设计、评分链路或数据覆盖诊断。")
    return "\n".join(lines) + "\n"


def _report_date_from(evaluated: pd.DataFrame, source_path: Path) -> str:
    if "date" in evaluated.columns and not evaluated.empty:
        parsed = pd.to_datetime(evaluated["date"], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().date().isoformat()
    return source_path.stem.replace("_replay_evaluated", "")


def write_scoring_v3_diagnostics_outputs(evaluated_path: str | Path, output_dir: str | Path, config: StrategyConfig | None = None) -> dict[str, Path]:
    source_path, output_path = Path(evaluated_path), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluated = pd.read_csv(source_path, dtype={"code": str})
    config = config or StrategyConfig()
    report_date = _report_date_from(evaluated, source_path)
    v1_scored = score_experimental_layer(evaluated, config)
    factor_diagnostics = build_factor_effectiveness_diagnostics(v1_scored)
    v2_scored = score_experimental_v2_layer(v1_scored, config)
    v2_diagnostics = build_scoring_robustness_diagnostics_for_profile(v2_scored, config, EXPERIMENTAL_V2_PROFILE)
    v3_scored = score_experimental_v3_layer(v2_scored, config)
    v3_diagnostics = build_scoring_robustness_diagnostics_for_profile(v3_scored, config, EXPERIMENTAL_V3_PROFILE)
    v3_diagnostics["v2_v3_focused_overlap"] = _build_v2_v3_focused_overlap(v2_scored, v3_scored)
    paths = {
        "factor_report": output_path / f"{report_date}_factor_effectiveness_report.md", "factor_by_factor": output_path / f"{report_date}_factor_effectiveness_by_factor.csv", "factor_by_date": output_path / f"{report_date}_factor_effectiveness_by_date.csv", "v3_report": output_path / f"{report_date}_v3_diagnostics_report.md", "v3_summary": output_path / f"{report_date}_v3_robustness_summary.md", "v3_by_date": output_path / f"{report_date}_v3_robustness_by_date.csv", "v3_by_horizon": output_path / f"{report_date}_v3_robustness_by_horizon.csv", "v3_quantile_monotonicity": output_path / f"{report_date}_v3_robustness_quantile_monotonicity.csv", "v3_focused_overlap": output_path / f"{report_date}_v3_robustness_focused_overlap.csv", "v3_flags": output_path / f"{report_date}_v3_robustness_flags.csv",
    }
    paths["factor_report"].write_text(build_factor_effectiveness_report(factor_diagnostics, source_path.name), encoding="utf-8")
    _with_csv_safety_notices(factor_diagnostics["by_factor"]).to_csv(
        paths["factor_by_factor"], index=False
    )
    _with_csv_safety_notices(factor_diagnostics["by_date"]).to_csv(
        paths["factor_by_date"], index=False
    )
    paths["v3_report"].write_text(build_v3_diagnostics_report(evaluated, v2_diagnostics, v3_diagnostics, source_path.name), encoding="utf-8")
    paths["v3_summary"].write_text(build_scoring_robustness_report(v3_diagnostics, source_path.name) + _gate_sync_section(v2_diagnostics, v3_diagnostics), encoding="utf-8")
    for path_key, diagnostic_key in (("v3_by_date", "by_date"), ("v3_by_horizon", "by_horizon"), ("v3_quantile_monotonicity", "quantile_monotonicity"), ("v3_focused_overlap", "focused_overlap"), ("v3_flags", "flags")):
        _with_csv_safety_notices(v3_diagnostics[diagnostic_key]).to_csv(
            paths[path_key], index=False
        )
    return paths
