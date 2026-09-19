from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import score_experimental_layer

RETURN_COLUMN_PATTERN = re.compile(r"^forward_return_(\d+)d$")

RESEARCH_NOTICE = "HISTORICAL RESEARCH ONLY / 历史研究用途"
RECOMMENDATION_NOTICE = "NOT A RECOMMENDATION / 不构成投资建议"
SCOPE_NOTICE = "LIMITED CACHED REPLAY / 有限缓存池回放"


@dataclass(frozen=True)
class RobustnessScoreProfile:
    name: str
    total_score_column: str
    tier_column: str
    requires_experimental_v1: bool = False
    requires_experimental_v2: bool = False
    requires_experimental_v3: bool = False


PRODUCTION_PROFILE = RobustnessScoreProfile("production", "total_score", "tier")
EXPERIMENTAL_V1_PROFILE = RobustnessScoreProfile(
    "experimental_v1",
    "experimental_total_score",
    "experimental_tier",
    requires_experimental_v1=True,
)
EXPERIMENTAL_V2_PROFILE = RobustnessScoreProfile(
    "experimental_v2",
    "experimental_v2_total_score",
    "experimental_v2_tier",
    requires_experimental_v2=True,
)

EXPERIMENTAL_V3_PROFILE = RobustnessScoreProfile(
    "experimental_v3",
    "experimental_v3_total_score",
    "experimental_v3_tier",
    requires_experimental_v3=True,
)
REQUIRED_COLUMNS = [
    "date",
    "code",
    "tier",
    "medium_term_score",
    "timing_score",
    "total_score",
    "signal_labels",
    "return_20d",
    "return_60d",
    "drawdown_60d",
    "volume_ratio_5_20",
    "above_ma20",
    "ma_bullish",
]

BY_DATE_COLUMNS = [
    "date",
    "horizon",
    "production_focused_sample_count",
    "production_candidate_sample_count",
    "production_focused_avg_return",
    "production_candidate_avg_return",
    "production_focused_win_rate",
    "production_candidate_win_rate",
    "production_focused_minus_candidate_avg_return",
    "production_focused_minus_candidate_win_rate",
    "experimental_focused_sample_count",
    "experimental_candidate_sample_count",
    "experimental_focused_avg_return",
    "experimental_candidate_avg_return",
    "experimental_focused_win_rate",
    "experimental_candidate_win_rate",
    "experimental_focused_minus_candidate_avg_return",
    "experimental_focused_minus_candidate_win_rate",
    "avg_delta_improvement",
    "win_delta_improvement",
    "experimental_better_avg",
    "experimental_better_win",
    "score_profile",
    "research_notice",
    "recommendation_notice",
    "scope_notice",
]
BY_HORIZON_COLUMNS = [
    "horizon",
    "date_count",
    "evaluable_date_count",
    "avg_comparable_date_count",
    "win_comparable_date_count",
    "avg_delta_improvement_mean",
    "avg_delta_improvement_median",
    "win_delta_improvement_mean",
    "win_delta_improvement_median",
    "experimental_better_avg_date_rate",
    "experimental_better_win_date_rate",
    "production_worst_delta_avg",
    "experimental_worst_delta_avg",
    "experimental_overall_delta_avg",
    "experimental_overall_negative",
    "score_profile",
    "research_notice",
    "recommendation_notice",
    "scope_notice",
]
QUANTILE_MONOTONICITY_COLUMNS = [
    "date",
    "horizon",
    "score_type",
    "q1_avg_return",
    "q3_avg_return",
    "q4_avg_return",
    "q4_minus_q1",
    "q4_minus_q3",
    "q4_below_q1",
    "q4_below_q3",
    "score_profile",
    "research_notice",
    "recommendation_notice",
    "scope_notice",
]
FOCUSED_OVERLAP_COLUMNS = [
    "date",
    "production_focused_count",
    "experimental_focused_count",
    "focused_overlap_count",
    "focused_overlap_rate",
    "score_profile",
    "research_notice",
    "recommendation_notice",
    "scope_notice",
]
FLAGS_COLUMNS = [
    "flag_type",
    "severity",
    "date",
    "horizon",
    "message",
    "score_profile",
    "research_notice",
    "recommendation_notice",
    "scope_notice",
]


def _return_columns(evaluated: pd.DataFrame) -> list[tuple[int, str]]:
    return sorted(
        (int(match.group(1)), str(column))
        for column in evaluated.columns
        if (match := RETURN_COLUMN_PATTERN.match(str(column)))
    )


def _validate_columns(evaluated: pd.DataFrame, profile: RobustnessScoreProfile | None = None) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in evaluated.columns]
    if profile is not None:
        for column in (profile.total_score_column, profile.tier_column):
            if column not in evaluated.columns:
                missing.append(column)
    if not _return_columns(evaluated):
        missing.append("forward_return_*d")
    if missing:
        raise ValueError(f"Missing required scoring robustness columns: {missing}")


def _with_research_notices(
    data: pd.DataFrame,
    columns: list[str],
    profile: RobustnessScoreProfile,
) -> pd.DataFrame:
    result = data.copy()
    result["score_profile"] = profile.name
    result["research_notice"] = RESEARCH_NOTICE
    result["recommendation_notice"] = RECOMMENDATION_NOTICE
    result["scope_notice"] = SCOPE_NOTICE
    return result.reindex(columns=columns)


def _to_long_returns(
    date_level_scores: pd.DataFrame,
    profile: RobustnessScoreProfile,
) -> pd.DataFrame:
    records = []
    for _, row in date_level_scores.iterrows():
        for horizon, column in _return_columns(date_level_scores):
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                records.append(
                    {
                        "_evaluation_id": int(row["_evaluation_id"]),
                        "date": str(row["date"]),
                        "code": str(row["code"]),
                        "tier": row["tier"],
                        "experimental_tier": row[profile.tier_column],
                        "total_score": row["total_score"],
                        "experimental_total_score": row[profile.total_score_column],
                        "horizon": horizon,
                        "forward_return": float(value),
                    }
                )
    return pd.DataFrame.from_records(records)


def _tier_metrics(group: pd.DataFrame, tier_column: str, prefix: str) -> dict[str, float | int]:
    focused = group.loc[group[tier_column].eq("focused"), "forward_return"]
    candidate = group.loc[group[tier_column].eq("candidate"), "forward_return"]
    focused_avg = float(focused.mean()) if not focused.empty else float("nan")
    candidate_avg = float(candidate.mean()) if not candidate.empty else float("nan")
    focused_win = float((focused > 0).mean()) if not focused.empty else float("nan")
    candidate_win = float((candidate > 0).mean()) if not candidate.empty else float("nan")
    return {
        f"{prefix}_focused_sample_count": int(focused.count()),
        f"{prefix}_candidate_sample_count": int(candidate.count()),
        f"{prefix}_focused_avg_return": focused_avg,
        f"{prefix}_candidate_avg_return": candidate_avg,
        f"{prefix}_focused_win_rate": focused_win,
        f"{prefix}_candidate_win_rate": candidate_win,
        f"{prefix}_focused_minus_candidate_avg_return": focused_avg - candidate_avg,
        f"{prefix}_focused_minus_candidate_win_rate": focused_win - candidate_win,
    }


def _build_by_date(long_returns: pd.DataFrame, profile: RobustnessScoreProfile) -> pd.DataFrame:
    if long_returns.empty:
        return pd.DataFrame(columns=BY_DATE_COLUMNS)
    rows = []
    for (date_value, horizon), group in long_returns.groupby(["date", "horizon"], dropna=False):
        row: dict[str, object] = {"date": str(date_value), "horizon": int(horizon)}
        row.update(_tier_metrics(group, "tier", "production"))
        row.update(_tier_metrics(group, "experimental_tier", "experimental"))
        row["avg_delta_improvement"] = (
            row["experimental_focused_minus_candidate_avg_return"]
            - row["production_focused_minus_candidate_avg_return"]
        )
        row["win_delta_improvement"] = (
            row["experimental_focused_minus_candidate_win_rate"]
            - row["production_focused_minus_candidate_win_rate"]
        )
        row["experimental_better_avg"] = (
            pd.NA if pd.isna(row["avg_delta_improvement"]) else bool(row["avg_delta_improvement"] > 0)
        )
        row["experimental_better_win"] = (
            pd.NA if pd.isna(row["win_delta_improvement"]) else bool(row["win_delta_improvement"] > 0)
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in ("experimental_better_avg", "experimental_better_win"):
        result[column] = pd.array(result[column], dtype="boolean")
    return _with_research_notices(result, BY_DATE_COLUMNS, profile)


def _nullable_boolean_rate(values: pd.Series) -> float:
    mean_value = values.astype("Float64").mean()
    return float(mean_value) if pd.notna(mean_value) else float("nan")


def _build_by_horizon(by_date: pd.DataFrame, profile: RobustnessScoreProfile) -> pd.DataFrame:
    if by_date.empty:
        return pd.DataFrame(columns=BY_HORIZON_COLUMNS)
    rows = []
    for horizon, group in by_date.groupby("horizon", dropna=False):
        avg_comparable = group["avg_delta_improvement"].dropna()
        win_comparable = group["win_delta_improvement"].dropna()
        experimental_delta = group["experimental_focused_minus_candidate_avg_return"]
        experimental_mean = experimental_delta.mean()
        rows.append(
            {
                "horizon": int(horizon),
                "date_count": int(avg_comparable.shape[0]),
                "evaluable_date_count": int(group["date"].nunique()),
                "avg_comparable_date_count": int(avg_comparable.shape[0]),
                "win_comparable_date_count": int(win_comparable.shape[0]),
                "avg_delta_improvement_mean": float(avg_comparable.mean()),
                "avg_delta_improvement_median": float(avg_comparable.median()),
                "win_delta_improvement_mean": float(win_comparable.mean()),
                "win_delta_improvement_median": float(win_comparable.median()),
                "experimental_better_avg_date_rate": _nullable_boolean_rate(
                    group.loc[avg_comparable.index, "experimental_better_avg"]
                ),
                "experimental_better_win_date_rate": _nullable_boolean_rate(
                    group.loc[win_comparable.index, "experimental_better_win"]
                ),
                "production_worst_delta_avg": float(
                    group["production_focused_minus_candidate_avg_return"].min()
                ),
                "experimental_worst_delta_avg": float(experimental_delta.min()),
                "experimental_overall_delta_avg": float(experimental_mean),
                "experimental_overall_negative": bool(pd.notna(experimental_mean) and experimental_mean < 0),
            }
        )
    return _with_research_notices(pd.DataFrame(rows), BY_HORIZON_COLUMNS, profile)


def _quantile_buckets(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unique_values = numeric.dropna().drop_duplicates().sort_values()
    bucket_count = min(4, len(unique_values))
    if bucket_count <= 1:
        return pd.Series("q1", index=values.index, dtype="string").where(numeric.notna(), "unknown")
    labels = {
        2: ["q1", "q4"],
        3: ["q1", "q3", "q4"],
        4: ["q1", "q2", "q3", "q4"],
    }[bucket_count]
    buckets_by_value = pd.Series(
        pd.qcut(unique_values, q=bucket_count, labels=labels).astype("string").to_numpy(),
        index=unique_values.to_numpy(),
    )
    return numeric.map(buckets_by_value).astype("string").fillna("unknown")


def _experimental_score_type(profile: RobustnessScoreProfile) -> str:
    if profile.name == "experimental_v1":
        return "experimental_total_score"
    return f"{profile.name}_total_score"


def _build_quantile_monotonicity(
    long_returns: pd.DataFrame,
    date_level_scores: pd.DataFrame,
    profile: RobustnessScoreProfile,
) -> pd.DataFrame:
    if long_returns.empty:
        return pd.DataFrame(columns=QUANTILE_MONOTONICITY_COLUMNS)
    rows = []
    score_columns = {
        "production_total_score": "total_score",
        _experimental_score_type(profile): profile.total_score_column,
    }
    for score_type, score_column in score_columns.items():
        bucket_map = date_level_scores[["_evaluation_id", "date", score_column]].copy()
        bucket_map["date"] = bucket_map["date"].astype(str)
        bucket_map["bucket"] = bucket_map.groupby("date", dropna=False)[score_column].transform(
            _quantile_buckets
        )
        scored = long_returns.merge(
            bucket_map[["_evaluation_id", "date", "bucket"]],
            on=["_evaluation_id", "date"],
            how="left",
            validate="many_to_one",
        )
        for (date_value, horizon), group in scored.groupby(["date", "horizon"], dropna=False):
            bucket_returns = group.groupby("bucket", dropna=False)["forward_return"].mean()
            q1 = float(bucket_returns.get("q1", float("nan")))
            q3 = float(bucket_returns.get("q3", float("nan")))
            q4 = float(bucket_returns.get("q4", float("nan")))
            rows.append(
                {
                    "date": str(date_value),
                    "horizon": int(horizon),
                    "score_type": score_type,
                    "q1_avg_return": q1,
                    "q3_avg_return": q3,
                    "q4_avg_return": q4,
                    "q4_minus_q1": q4 - q1,
                    "q4_minus_q3": q4 - q3,
                    "q4_below_q1": bool(q4 < q1),
                    "q4_below_q3": bool(q4 < q3),
                }
            )
    return _with_research_notices(pd.DataFrame(rows), QUANTILE_MONOTONICITY_COLUMNS, profile)


def _build_focused_overlap(
    scored: pd.DataFrame,
    profile: RobustnessScoreProfile,
) -> pd.DataFrame:
    rows = []
    for date_value, group in scored.groupby("date", dropna=False):
        production_codes = set(group.loc[group["tier"].eq("focused"), "code"].astype(str))
        experimental_codes = set(group.loc[group[profile.tier_column].eq("focused"), "code"].astype(str))
        denominator = len(production_codes | experimental_codes)
        rows.append(
            {
                "date": str(date_value),
                "production_focused_count": len(production_codes),
                "experimental_focused_count": len(experimental_codes),
                "focused_overlap_count": len(production_codes & experimental_codes),
                "focused_overlap_rate": len(production_codes & experimental_codes) / denominator if denominator else 0.0,
            }
        )
    return _with_research_notices(pd.DataFrame(rows), FOCUSED_OVERLAP_COLUMNS, profile)


def _build_flags(
    by_date: pd.DataFrame,
    by_horizon: pd.DataFrame,
    quantile_monotonicity: pd.DataFrame,
    profile: RobustnessScoreProfile,
) -> pd.DataFrame:
    rows = []
    for _, row in by_horizon.iterrows():
        horizon = int(row["horizon"])
        improvement_rate = row["experimental_better_avg_date_rate"]
        if pd.notna(improvement_rate) and improvement_rate < 0.5:
            rows.append(
                {
                    "flag_type": "low_improvement_rate",
                    "severity": "warning",
                    "date": "",
                    "horizon": horizon,
                    "message": f"{horizon}d 实验评分平均收益改善日期占比低于 50%，不支持直接升级为生产评分。",
                }
            )
        if bool(row["experimental_overall_negative"]):
            rows.append(
                {
                    "flag_type": "negative_experimental_delta",
                    "severity": "warning",
                    "date": "",
                    "horizon": horizon,
                    "message": f"{horizon}d 实验 focused 相对 candidate 的平均差仍为负。",
                }
            )
        comparable = by_date.loc[
            by_date["horizon"].eq(horizon)
            & by_date["experimental_focused_minus_candidate_avg_return"].notna()
        ]
        if not comparable.empty:
            worst = comparable.loc[comparable["experimental_focused_minus_candidate_avg_return"].idxmin()]
            if worst["experimental_focused_minus_candidate_avg_return"] < 0:
                rows.append(
                    {
                        "flag_type": "worst_date_risk",
                        "severity": "warning",
                        "date": str(worst["date"]),
                        "horizon": horizon,
                        "message": f"{worst['date']} 的 {horizon}d 实验 focused 相对 candidate 差值为该 horizon 最低且为负。",
                    }
                )
    for _, row in quantile_monotonicity.iterrows():
        if bool(row["q4_below_q1"]) or bool(row["q4_below_q3"]):
            rows.append(
                {
                    "flag_type": "quantile_reversal",
                    "severity": "warning",
                    "date": str(row["date"]),
                    "horizon": int(row["horizon"]),
                    "message": f"{row['score_type']} 在 {row['date']} 的 {int(row['horizon'])}d 出现高分组弱于低分组或中高分组。",
                }
            )
    return _with_research_notices(pd.DataFrame(rows), FLAGS_COLUMNS, profile)


def _prepare_profile_scores(
    evaluated: pd.DataFrame,
    config: StrategyConfig,
    profile: RobustnessScoreProfile,
) -> pd.DataFrame:
    if profile.requires_experimental_v1:
        return score_experimental_layer(evaluated, config)
    _validate_columns(evaluated, profile)
    return evaluated.copy()


def build_scoring_robustness_diagnostics_for_profile(
    evaluated: pd.DataFrame,
    config: StrategyConfig,
    profile: RobustnessScoreProfile,
) -> dict[str, pd.DataFrame]:
    _validate_columns(evaluated)
    scored = _prepare_profile_scores(evaluated, config, profile)
    date_level_scores = scored.copy().reset_index(drop=True)
    _validate_columns(date_level_scores, profile)
    date_level_scores["_evaluation_id"] = range(len(date_level_scores))
    long_returns = _to_long_returns(date_level_scores, profile)
    by_date = _build_by_date(long_returns, profile)
    by_horizon = _build_by_horizon(by_date, profile)
    quantile_monotonicity = _build_quantile_monotonicity(long_returns, date_level_scores, profile)
    focused_overlap = _build_focused_overlap(date_level_scores, profile)
    return {
        "by_date": by_date,
        "by_horizon": by_horizon,
        "quantile_monotonicity": quantile_monotonicity,
        "focused_overlap": focused_overlap,
        "flags": _build_flags(by_date, by_horizon, quantile_monotonicity, profile),
    }


def build_scoring_robustness_diagnostics(
    evaluated: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, pd.DataFrame]:
    return build_scoring_robustness_diagnostics_for_profile(evaluated, config, EXPERIMENTAL_V1_PROFILE)


def _format_percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(numeric) else f"{numeric:.2%}"


def _markdown_horizon_table(by_horizon: pd.DataFrame) -> list[str]:
    lines = [
        "| Horizon | Evaluable dates | Comparable avg dates | Avg improvement | Better avg date rate | Experimental overall delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in by_horizon.sort_values("horizon").iterrows():
        lines.append(
            f"| {int(row['horizon'])}d | {int(row['evaluable_date_count'])} | "
            f"{int(row['avg_comparable_date_count'])} | "
            f"{_format_percent(row['avg_delta_improvement_mean'])} | "
            f"{_format_percent(row['experimental_better_avg_date_rate'])} | "
            f"{_format_percent(row['experimental_overall_delta_avg'])} |"
        )
    return lines


def _worst_date_lines(by_date: pd.DataFrame) -> list[str]:
    lines = ["## 最差日期", ""]
    if by_date.empty:
        return lines + ["- 没有可评估的最差日期。"]
    for horizon, group in by_date.groupby("horizon", dropna=False):
        comparable = group.loc[group["experimental_focused_minus_candidate_avg_return"].notna()]
        if comparable.empty:
            lines.append(f"- {int(horizon)}d：没有 production/experimental 都可比较的日期。")
            continue
        worst = comparable.loc[comparable["experimental_focused_minus_candidate_avg_return"].idxmin()]
        lines.append(
            f"- {int(horizon)}d：{worst['date']}，实验 focused 相对 candidate 差值 "
            f"{_format_percent(worst['experimental_focused_minus_candidate_avg_return'])}。"
        )
    return lines


def _quantile_summary_lines(quantile_monotonicity: pd.DataFrame) -> list[str]:
    lines = ["## 分位组单调性", ""]
    if quantile_monotonicity.empty:
        return lines + ["- 没有可用分位组收益数据。"]
    reversals = quantile_monotonicity.loc[
        quantile_monotonicity["q4_below_q1"] | quantile_monotonicity["q4_below_q3"]
    ]
    lines.append(f"- 检查记录数：{len(quantile_monotonicity)}；高分组反向记录数：{len(reversals)}。")
    if not reversals.empty:
        details = ", ".join(
            f"{row.date}/{int(row.horizon)}d/{row.score_type}" for row in reversals.head(20).itertuples()
        )
        lines.append(f"- 反向明细（最多 20 条）：{details}")
    return lines


def _focused_overlap_lines(overlap: pd.DataFrame) -> list[str]:
    lines = ["## Focused 股票池重合", ""]
    if overlap.empty:
        return lines + ["- 没有 focused overlap 数据。"]
    lines.extend(
        [
            f"- 平均重合率：{_format_percent(overlap['focused_overlap_rate'].mean())}",
            f"- 最低重合率：{_format_percent(overlap['focused_overlap_rate'].min())}",
            "- 低重合日期（低于 50%）：",
        ]
    )
    low_overlap = overlap.loc[overlap["focused_overlap_rate"] < 0.5]
    if low_overlap.empty:
        lines.append("  - 无。")
    else:
        for row in low_overlap.sort_values("focused_overlap_rate").itertuples():
            lines.append(
                f"  - {row.date}：{_format_percent(row.focused_overlap_rate)} "
                f"({int(row.focused_overlap_count)} 个重合，生产 {int(row.production_focused_count)}，"
                f"实验 {int(row.experimental_focused_count)})。"
            )
    return lines


def _stage_judgment(by_horizon: pd.DataFrame, quantile_monotonicity: pd.DataFrame) -> str:
    if by_horizon.empty:
        return "可评估样本不足，当前证据不支持进入 v0.6.6 权重校准；应先补充历史缓存回放样本并检查因子设计。"
    horizon_count = len(by_horizon)
    negative_count = int(by_horizon["experimental_overall_negative"].sum())
    weak_rate_count = int(
        (
            by_horizon["experimental_better_avg_date_rate"].notna()
            & (by_horizon["experimental_better_avg_date_rate"] < 0.5)
        ).sum()
    )
    reversal_count = int(
        (quantile_monotonicity["q4_below_q1"] | quantile_monotonicity["q4_below_q3"]).sum()
    )
    weak_evidence = negative_count * 2 >= horizon_count or weak_rate_count * 2 >= horizon_count or reversal_count >= horizon_count
    if weak_evidence:
        return (
            "当前缓存回放证据不支持进入 v0.6.6 权重校准，也不支持升级生产评分；"
            f"其中负 experimental delta horizon 为 {negative_count}/{horizon_count}，"
            f"改善率偏弱 horizon 为 {weak_rate_count}/{horizon_count}，高分组反向记录为 {reversal_count}。"
            "建议先回到因子设计与评分链路诊断。"
        )
    return "当前缓存回放中未触发既定的弱证据门槛；但本通用稳健性报告不单独支持进入权重校准，是否进入下一步必须以专门 gate 报告为准。生产评分保持不变，仍不得据此生成交易建议。"



def _diagnostics_profile_name(diagnostics: dict[str, pd.DataFrame]) -> str:
    for key in ("by_horizon", "by_date", "quantile_monotonicity", "focused_overlap", "flags"):
        frame = diagnostics.get(key, pd.DataFrame())
        if not frame.empty and "score_profile" in frame.columns:
            values = frame["score_profile"].dropna()
            if not values.empty:
                return str(values.iloc[0])
    return "unknown"
def build_scoring_robustness_report(diagnostics: dict[str, pd.DataFrame], source_name: str) -> str:
    by_date = diagnostics["by_date"]
    by_horizon = diagnostics["by_horizon"]
    quantile_monotonicity = diagnostics["quantile_monotonicity"]
    flags = diagnostics["flags"]
    overlap = diagnostics["focused_overlap"]
    profile_name = _diagnostics_profile_name(diagnostics)
    input_date_count = int(overlap["date"].nunique()) if not overlap.empty else 0
    evaluable_date_count = int(by_date["date"].nunique()) if not by_date.empty else 0
    lines = [
        "# 多日期评分稳健性诊断报告",
        "",
        "ROBUSTNESS DIAGNOSTICS / 稳健性诊断",
        RESEARCH_NOTICE,
        RECOMMENDATION_NOTICE,
        SCOPE_NOTICE,
        "",
        f"- source: {source_name}",
        f"- score profile: {profile_name}",
        f"- input replay date count: {input_date_count}",
        f"- evaluable replay date count: {evaluable_date_count}",
        "- scope: local cached replay outputs only; production scores and tiers remain unchanged.",
        "",
    ]
    if evaluable_date_count < 2:
        lines.extend(["## 样本日期不足", "", "当前可评估 replay date 少于 2 个，不能判断多日期稳健性；本报告仅保留可审计诊断结果。", ""])
    lines.extend(["## Horizon 稳定性摘要", ""])
    lines.extend(_markdown_horizon_table(by_horizon) if not by_horizon.empty else ["没有可用 forward return，无法计算稳定性。"])
    lines.extend([""])
    lines.extend(_worst_date_lines(by_date))
    lines.extend([""])
    lines.extend(_quantile_summary_lines(quantile_monotonicity))
    lines.extend(["", "## 风险标记", ""])
    if flags.empty:
        lines.append("- 未发现自动规则标记；这不代表可以实盘，只表示当前规则未触发。")
    else:
        for _, row in flags.head(20).iterrows():
            date_text = f"{row['date']} " if str(row["date"]) else ""
            lines.append(f"- [{row['severity']}] {date_text}{int(row['horizon'])}d: {row['message']}")
    lines.extend([""])
    lines.extend(_focused_overlap_lines(overlap))
    lines.extend(["", "## 下一阶段判断", "", _stage_judgment(by_horizon, quantile_monotonicity)])
    return "\n".join(lines) + "\n"


def _report_date_from(evaluated: pd.DataFrame, source_path: Path) -> str:
    if "date" in evaluated.columns and not evaluated.empty:
        parsed = pd.to_datetime(evaluated["date"], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().date().isoformat()
    return source_path.stem.replace("_replay_evaluated", "")


def write_scoring_robustness_outputs(
    evaluated_path: str | Path,
    output_dir: str | Path,
    config: StrategyConfig,
) -> dict[str, Path]:
    source_path, output_path = Path(evaluated_path), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluated = pd.read_csv(source_path, dtype={"code": str})
    report_date = _report_date_from(evaluated, source_path)
    diagnostics = build_scoring_robustness_diagnostics(evaluated, config)
    paths = {
        "summary": output_path / f"{report_date}_robustness_summary.md",
        "by_date": output_path / f"{report_date}_robustness_by_date.csv",
        "by_horizon": output_path / f"{report_date}_robustness_by_horizon.csv",
        "quantile_monotonicity": output_path / f"{report_date}_robustness_quantile_monotonicity.csv",
        "focused_overlap": output_path / f"{report_date}_robustness_focused_overlap.csv",
        "flags": output_path / f"{report_date}_robustness_flags.csv",
    }
    paths["summary"].write_text(build_scoring_robustness_report(diagnostics, source_path.name), encoding="utf-8")
    for key in ("by_date", "by_horizon", "quantile_monotonicity", "focused_overlap", "flags"):
        diagnostics[key].to_csv(paths[key], index=False)
    return paths
