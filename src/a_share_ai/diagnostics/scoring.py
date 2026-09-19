from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from a_share_ai.models import StrategyConfig
from a_share_ai.strategies.experimental import score_experimental_layer

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
]

RETURN_COLUMN_PATTERN = re.compile(r"^forward_return_(\d+)d$")

TIER_DELTA_COLUMNS = [
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



EXPERIMENTAL_OVERLAP_COLUMNS = [
    "date",
    "production_focused_count",
    "experimental_focused_count",
    "focused_overlap_count",
    "focused_overlap_rate",
]
def build_scoring_diagnostics(evaluated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _validate_columns(evaluated)
    long_returns = _to_long_returns(evaluated, expand_signal_labels=False)
    return {
        "tier_delta": _build_tier_delta(long_returns),
        "score_factor_buckets": _build_factor_buckets(evaluated, long_returns),
        "signal_label_diagnostics": _build_signal_label_diagnostics(
            _to_long_returns(evaluated, expand_signal_labels=True)
        ),
        "signal_combo_diagnostics": _build_signal_combo_diagnostics(
            _to_long_returns(evaluated, expand_signal_labels=False, combo_labels=True)
        ),
        "bad_replay_dates": _build_bad_replay_dates(long_returns),
    }


def build_experimental_scoring_diagnostics(
    evaluated: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, pd.DataFrame]:
    _validate_columns(evaluated)
    experimental = score_experimental_layer(evaluated, config)
    long_returns = _to_long_returns(
        experimental.rename(columns={"tier": "production_tier"}).assign(
            tier=experimental["experimental_tier"]
        ),
        expand_signal_labels=False,
    )
    return {
        "experimental_tier_delta": _build_tier_delta(long_returns),
        "experimental_factor_buckets": _build_factor_buckets_for_columns(
            experimental,
            long_returns,
            EXPERIMENTAL_FACTOR_COLUMNS,
        ),
        "experimental_overlap": _build_experimental_overlap(experimental),
    }

def _validate_columns(evaluated: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in evaluated.columns]
    if not _return_columns(evaluated):
        missing.append("forward_return_*d")
    if missing:
        raise ValueError(f"Missing required scoring diagnostics columns: {missing}")


def _return_columns(evaluated: pd.DataFrame) -> list[tuple[int, str]]:
    matches = []
    for column in evaluated.columns:
        match = RETURN_COLUMN_PATTERN.match(str(column))
        if match:
            matches.append((int(match.group(1)), str(column)))
    return sorted(matches)


def _to_long_returns(evaluated: pd.DataFrame, expand_signal_labels: bool, combo_labels: bool = False) -> pd.DataFrame:
    records = []
    for evaluation_id, (_, row) in enumerate(evaluated.iterrows()):
        labels = _split_signal_labels(row["signal_labels"])
        label_values = labels if expand_signal_labels else ["all"]
        if combo_labels:
            label_values = [" | ".join(labels) if labels else "unknown"]
        for horizon, column in _return_columns(evaluated):
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            for label in label_values:
                records.append(
                    {
                        "_evaluation_id": evaluation_id,
                        "date": row["date"],
                        "code": row["code"],
                        "tier": row["tier"],
                        "medium_term_score": row["medium_term_score"],
                        "timing_score": row["timing_score"],
                        "total_score": row["total_score"],
                        "return_20d": row["return_20d"],
                        "return_60d": row["return_60d"],
                        "drawdown_60d": row["drawdown_60d"],
                        "volume_ratio_5_20": row["volume_ratio_5_20"],
                        "signal_label": label,
                        "horizon": horizon,
                        "forward_return": float(value),
                    }
                )
    return pd.DataFrame.from_records(records)


FACTOR_COLUMNS = [
    "medium_term_score", "timing_score", "total_score", "return_20d",
    "return_60d", "drawdown_60d", "volume_ratio_5_20",
]


EXPERIMENTAL_FACTOR_COLUMNS = [
    "experimental_medium_term_score",
    "experimental_timing_score",
    "experimental_total_score",
]
FACTOR_BUCKET_COLUMNS = [
    "factor", "bucket_type", "bucket", "horizon", "sample_count", "avg_return",
    "median_return", "win_rate", "best_return", "worst_return",
]
SIGNAL_LABEL_COLUMNS = [
    "signal_label", "horizon", "sample_count", "avg_return", "median_return",
    "win_rate", "best_return", "worst_return", "focused_share",
]
SIGNAL_COMBO_COLUMNS = [
    "signal_combo", "horizon", "sample_count", "avg_return", "median_return",
    "win_rate", "best_return", "worst_return",
]
BAD_DATE_COLUMNS = [
    "date", "horizon", "sample_count", "avg_return", "median_return", "win_rate", "worst_return",
]


def _build_factor_buckets(evaluated: pd.DataFrame, long_returns: pd.DataFrame) -> pd.DataFrame:
    quantile_rows = _build_factor_buckets_for_columns(evaluated, long_returns, FACTOR_COLUMNS)
    if long_returns.empty:
        return quantile_rows
    fixed_data = long_returns.copy()
    fixed_data["factor"] = "total_score"
    fixed_data["bucket_type"] = "fixed"
    fixed_data["bucket"] = pd.cut(
        pd.to_numeric(fixed_data["total_score"], errors="coerce"),
        bins=[float("-inf"), 70, 80, 90, float("inf")],
        labels=["<70", "70-80", "80-90", ">=90"], right=False,
    ).astype("string").fillna("unknown")
    fixed_rows = _summarize(fixed_data, ["factor", "bucket_type", "bucket"])
    return pd.concat([quantile_rows, fixed_rows], ignore_index=True)


def _build_factor_buckets_for_columns(
    evaluated: pd.DataFrame,
    long_returns: pd.DataFrame,
    factor_columns: list[str],
) -> pd.DataFrame:
    if long_returns.empty:
        return pd.DataFrame(columns=FACTOR_BUCKET_COLUMNS)
    evaluated_factors = evaluated.reset_index(drop=True).copy()
    evaluated_factors["_evaluation_id"] = range(len(evaluated_factors))
    frames = []
    for factor in factor_columns:
        factor_buckets = evaluated_factors[["_evaluation_id", factor]].copy()
        factor_buckets["bucket"] = _quantile_buckets(factor_buckets[factor])
        factor_data = long_returns.merge(
            factor_buckets[["_evaluation_id", "bucket"]], on="_evaluation_id", how="left"
        )
        factor_data["factor"] = factor
        factor_data["bucket_type"] = "quantile"
        frames.append(_summarize(factor_data, ["factor", "bucket_type", "bucket"]))
    return pd.concat(frames, ignore_index=True)


def _build_experimental_overlap(experimental: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date_value, group in experimental.groupby("date", dropna=False):
        production_codes = set(group.loc[group["tier"].eq("focused"), "code"])
        experimental_codes = set(group.loc[group["experimental_tier"].eq("focused"), "code"])
        overlap_count = len(production_codes & experimental_codes)
        denominator = len(production_codes | experimental_codes)
        rows.append(
            {
                "date": date_value,
                "production_focused_count": len(production_codes),
                "experimental_focused_count": len(experimental_codes),
                "focused_overlap_count": overlap_count,
                "focused_overlap_rate": overlap_count / denominator if denominator else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=EXPERIMENTAL_OVERLAP_COLUMNS)

def _quantile_buckets(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unique_values = numeric.dropna().drop_duplicates().sort_values()
    bucket_count = min(4, len(unique_values))
    if bucket_count <= 1:
        return pd.Series("q1", index=values.index, dtype="string").where(numeric.notna(), "unknown")
    labels = [f"q{i}" for i in range(1, bucket_count + 1)]
    buckets_by_value = pd.Series(
        pd.qcut(unique_values, q=bucket_count, labels=labels).astype("string").to_numpy(),
        index=unique_values.to_numpy(),
    )
    return numeric.map(buckets_by_value).astype("string").fillna("unknown")

def _build_signal_label_diagnostics(label_returns: pd.DataFrame) -> pd.DataFrame:
    if label_returns.empty:
        return pd.DataFrame(columns=SIGNAL_LABEL_COLUMNS)
    summary = _summarize(label_returns, ["signal_label"])
    focused_share = (
        label_returns.assign(is_focused=label_returns["tier"].eq("focused").astype(float))
        .groupby(["signal_label", "horizon"], dropna=False)["is_focused"]
        .mean().reset_index(name="focused_share")
    )
    return summary.merge(focused_share, on=["signal_label", "horizon"]).sort_values(
        ["horizon", "avg_return", "signal_label"]
    ).reset_index(drop=True)


def _build_signal_combo_diagnostics(combo_returns: pd.DataFrame) -> pd.DataFrame:
    if combo_returns.empty:
        return pd.DataFrame(columns=SIGNAL_COMBO_COLUMNS)
    renamed = combo_returns.rename(columns={"signal_label": "signal_combo"})
    return _summarize(renamed, ["signal_combo"]).sort_values(
        ["horizon", "avg_return", "signal_combo"]
    ).reset_index(drop=True)


def _build_bad_replay_dates(long_returns: pd.DataFrame) -> pd.DataFrame:
    if long_returns.empty:
        return pd.DataFrame(columns=BAD_DATE_COLUMNS)
    summary = _summarize(long_returns, ["date"])
    return summary.sort_values(["horizon", "avg_return", "date"]).reset_index(drop=True)

def _build_tier_delta(long_returns: pd.DataFrame) -> pd.DataFrame:
    if long_returns.empty:
        return pd.DataFrame(columns=TIER_DELTA_COLUMNS)

    summary = _summarize(long_returns, ["tier"])
    rows = []
    for horizon in sorted(summary["horizon"].unique()):
        horizon_summary = summary.loc[summary["horizon"].eq(horizon)].set_index("tier")
        focused = horizon_summary.loc["focused"] if "focused" in horizon_summary.index else None
        candidate = horizon_summary.loc["candidate"] if "candidate" in horizon_summary.index else None
        if focused is None or candidate is None:
            continue
        rows.append(
            {
                "horizon": int(horizon),
                "focused_sample_count": int(focused["sample_count"]),
                "candidate_sample_count": int(candidate["sample_count"]),
                "focused_avg_return": float(focused["avg_return"]),
                "candidate_avg_return": float(candidate["avg_return"]),
                "focused_median_return": float(focused["median_return"]),
                "candidate_median_return": float(candidate["median_return"]),
                "focused_win_rate": float(focused["win_rate"]),
                "candidate_win_rate": float(candidate["win_rate"]),
                "focused_minus_candidate_avg_return": float(focused["avg_return"] - candidate["avg_return"]),
                "focused_minus_candidate_win_rate": float(focused["win_rate"] - candidate["win_rate"]),
            }
        )
    return pd.DataFrame(rows, columns=TIER_DELTA_COLUMNS)


def _summarize(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouping = [*group_columns, "horizon"]
    grouped = data.groupby(grouping, dropna=False)["forward_return"]
    summary = grouped.agg(
        sample_count="count",
        avg_return="mean",
        median_return="median",
        best_return="max",
        worst_return="min",
    ).reset_index()
    win_rate = grouped.apply(lambda values: (values > 0).mean()).reset_index(name="win_rate")
    return summary.merge(win_rate, on=grouping).sort_values(grouping).reset_index(drop=True)


def _split_signal_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text.strip("[]").replace("'", "").replace('"', "")
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]

SAFETY_LINES = [
    "SCORING DIAGNOSTICS / 评分诊断",
    "HISTORICAL RESEARCH ONLY / 历史研究用途",
    "NOT A RECOMMENDATION / 不构成投资建议",
    "LIMITED CACHED REPLAY / 有限缓存池回放",
]


EXPERIMENTAL_CSV_SAFETY_METADATA = {
    "research_notice": SAFETY_LINES[1],
    "recommendation_notice": SAFETY_LINES[2],
    "scope_notice": SAFETY_LINES[3],
}

def write_scoring_diagnostics_outputs(
    evaluated_path: str | Path,
    output_dir: str | Path,
    *,
    include_experimental: bool = False,
    strategy_config: StrategyConfig | None = None,
) -> dict[str, Path]:
    source_path = Path(evaluated_path)
    evaluated = pd.read_csv(source_path, dtype={"code": str})
    diagnostics = build_scoring_diagnostics(evaluated)
    report_date = _infer_report_date(source_path, evaluated)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_path / f"{report_date}_scoring_diagnostics.md",
        "tier_delta": output_path / f"{report_date}_tier_delta.csv",
        "score_factor_buckets": output_path / f"{report_date}_score_factor_buckets.csv",
        "signal_label_diagnostics": output_path / f"{report_date}_signal_label_diagnostics.csv",
        "signal_combo_diagnostics": output_path / f"{report_date}_signal_combo_diagnostics.csv",
        "bad_replay_dates": output_path / f"{report_date}_bad_replay_dates.csv",
    }
    paths["report"].write_text(build_scoring_diagnostics_report(diagnostics, source_path.name), encoding="utf-8")
    for key in ["tier_delta", "score_factor_buckets", "signal_label_diagnostics", "signal_combo_diagnostics", "bad_replay_dates"]:
        diagnostics[key].to_csv(paths[key], index=False)
    if include_experimental:
        experimental = build_experimental_scoring_diagnostics(evaluated, strategy_config or StrategyConfig())
        experimental["production_tier_delta"] = diagnostics["tier_delta"]
        experimental["production_factor_buckets"] = diagnostics["score_factor_buckets"]
        experimental_paths = {
            "experimental_report": output_path / f"{report_date}_experimental_scoring_comparison.md",
            "experimental_tier_delta": output_path / f"{report_date}_experimental_tier_delta.csv",
            "experimental_factor_buckets": output_path / f"{report_date}_experimental_factor_buckets.csv",
            "experimental_overlap": output_path / f"{report_date}_experimental_overlap.csv",
        }
        experimental_paths["experimental_report"].write_text(build_experimental_scoring_comparison_report(experimental, source_path.name), encoding="utf-8")
        for key in ["experimental_tier_delta", "experimental_factor_buckets", "experimental_overlap"]:
            experimental[key].assign(**EXPERIMENTAL_CSV_SAFETY_METADATA).to_csv(experimental_paths[key], index=False)
        paths.update(experimental_paths)
    return paths



def build_experimental_scoring_comparison_report(
    diagnostics: dict[str, pd.DataFrame], source_name: str
) -> str:
    production_tier_delta = diagnostics["production_tier_delta"].set_index("horizon")
    experimental_tier_delta = diagnostics["experimental_tier_delta"].set_index("horizon")
    production_buckets = diagnostics["production_factor_buckets"]
    experimental_buckets = diagnostics["experimental_factor_buckets"]
    production_quantiles = production_buckets.loc[
        production_buckets["factor"].eq("total_score")
        & production_buckets["bucket_type"].eq("quantile")
    ].set_index(["horizon", "bucket"])
    experimental_quantiles = experimental_buckets.loc[
        experimental_buckets["factor"].eq("experimental_total_score")
        & experimental_buckets["bucket_type"].eq("quantile")
    ].set_index(["horizon", "bucket"])

    def format_optional_percent(row: pd.Series | None, column: str) -> str:
        return "n/a" if row is None else _format_percent(row[column])

    lines = [
        "# \u5b9e\u9a8c\u8bc4\u5206\u5bf9\u6bd4\u62a5\u544a",
        "",
        "EXPERIMENTAL SCORING COMPARISON / \u5b9e\u9a8c\u8bc4\u5206\u5bf9\u6bd4",
        *SAFETY_LINES[1:],
        "",
        f"- source: {source_name}",
        "- scope: local cached replay outputs only; does not change production tier.",
        "",
        "## Production vs experimental tier delta",
        "",
        "| Horizon | Production delta avg | Experimental delta avg | Production delta win | Experimental delta win |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for horizon in sorted(set(production_tier_delta.index) | set(experimental_tier_delta.index)):
        production_row = (
            production_tier_delta.loc[horizon]
            if horizon in production_tier_delta.index
            else None
        )
        experimental_row = (
            experimental_tier_delta.loc[horizon]
            if horizon in experimental_tier_delta.index
            else None
        )
        lines.append(
            "| {horizon}d | {production_avg} | {experimental_avg} | {production_win} | {experimental_win} |".format(
                horizon=int(horizon),
                production_avg=format_optional_percent(
                    production_row, "focused_minus_candidate_avg_return"
                ),
                experimental_avg=format_optional_percent(
                    experimental_row, "focused_minus_candidate_avg_return"
                ),
                production_win=format_optional_percent(
                    production_row, "focused_minus_candidate_win_rate"
                ),
                experimental_win=format_optional_percent(
                    experimental_row, "focused_minus_candidate_win_rate"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Production vs experimental total-score quantiles",
            "",
            "| Horizon | Quantile | Production avg return | Experimental avg return |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for horizon, bucket in sorted(set(production_quantiles.index) | set(experimental_quantiles.index)):
        production_row = (
            production_quantiles.loc[(horizon, bucket)]
            if (horizon, bucket) in production_quantiles.index
            else None
        )
        experimental_row = (
            experimental_quantiles.loc[(horizon, bucket)]
            if (horizon, bucket) in experimental_quantiles.index
            else None
        )
        lines.append(
            "| {horizon}d | {bucket} | {production_avg} | {experimental_avg} |".format(
                horizon=int(horizon),
                bucket=bucket,
                production_avg=format_optional_percent(production_row, "avg_return"),
                experimental_avg=format_optional_percent(experimental_row, "avg_return"),
            )
        )
    lines.extend(
        [
            "",
            "## Focused overlap by date",
            "",
            "| Date | Production focused | Experimental focused | Overlap | Overlap rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in diagnostics["experimental_overlap"].iterrows():
        lines.append(
            "| {date} | {production_count} | {experimental_count} | {overlap_count} | {overlap_rate} |".format(
                date=row["date"],
                production_count=int(row["production_focused_count"]),
                experimental_count=int(row["experimental_focused_count"]),
                overlap_count=int(row["focused_overlap_count"]),
                overlap_rate=_format_percent(row["focused_overlap_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Experimental factor buckets",
            "",
            f"- {len(experimental_buckets)} bucket summaries; see `*_experimental_factor_buckets.csv`.",
            "",
            "## \u9605\u8bfb\u63d0\u9192",
            "",
            "Historical cached-replay research only; production scores and tiers remain unchanged.",
            "Experimental scoring cannot produce trading instructions or automatic orders.",
        ]
    )
    return chr(10).join(lines)

def build_experimental_scoring_report(diagnostics: dict[str, pd.DataFrame], source_name: str) -> str:
    lines = [
        "# 实验评分对比报告", "", "EXPERIMENTAL SCORING COMPARISON / 实验评分对比", *SAFETY_LINES[1:], "",
        f"- source: {source_name}", "- scope: local cached replay outputs only; does not change production tier.", "",
        "## Experimental focused vs candidate", "",
        "| Horizon | Experimental focused avg | Experimental candidate avg | Delta avg | Experimental focused win | Experimental candidate win | Delta win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics["experimental_tier_delta"].iterrows():
        lines.append("| {horizon}d | {focused_avg} | {candidate_avg} | {delta_avg} | {focused_win} | {candidate_win} | {delta_win} |".format(
            horizon=int(row["horizon"]), focused_avg=_format_percent(row["focused_avg_return"]), candidate_avg=_format_percent(row["candidate_avg_return"]), delta_avg=_format_percent(row["focused_minus_candidate_avg_return"]), focused_win=_format_percent(row["focused_win_rate"]), candidate_win=_format_percent(row["candidate_win_rate"]), delta_win=_format_percent(row["focused_minus_candidate_win_rate"]),
        ))
    lines.extend(["", "## Experimental factor buckets", "", f"- {len(diagnostics['experimental_factor_buckets'])} bucket summaries; see `*_experimental_factor_buckets.csv`.", "", "## Focused overlap", "", f"- {len(diagnostics['experimental_overlap'])} date summaries; see `*_experimental_overlap.csv`.", "", "## 阅读提醒", "", "本报告只比较历史缓存回放中的实验评分，不改变正式评分、正式分层或任何交易动作。", "实验评分不能直接用于买卖、加仓、减仓或自动下单。"])
    return chr(10).join(lines)

def build_scoring_diagnostics_report(diagnostics: dict[str, pd.DataFrame], source_name: str) -> str:
    lines = [
        "# 评分与分层诊断报告",
        "",
        *SAFETY_LINES,
        "",
        f"- source: {source_name}",
        "- scope: local cached replay outputs only; not a full A-share market claim.",
        "",
        "## Focused vs Candidate",
        "",
        "| Horizon | Focused avg | Candidate avg | Delta avg | Focused win | Candidate win | Delta win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics["tier_delta"].iterrows():
        lines.append(
            "| {horizon}d | {focused_avg} | {candidate_avg} | {delta_avg} | {focused_win} | {candidate_win} | {delta_win} |".format(
                horizon=int(row["horizon"]),
                focused_avg=_format_percent(row["focused_avg_return"]),
                candidate_avg=_format_percent(row["candidate_avg_return"]),
                delta_avg=_format_percent(row["focused_minus_candidate_avg_return"]),
                focused_win=_format_percent(row["focused_win_rate"]),
                candidate_win=_format_percent(row["candidate_win_rate"]),
                delta_win=_format_percent(row["focused_minus_candidate_win_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Factor buckets",
            "",
            f"- {len(diagnostics['score_factor_buckets'])} bucket summaries; see the adjacent `*_score_factor_buckets.csv`.",
            "- Quantiles are assigned from de-duplicated evaluated factor values before return-horizon expansion.",
            "",
            "## Signal labels and combinations",
            "",
            f"- {len(diagnostics['signal_label_diagnostics'])} label summaries and {len(diagnostics['signal_combo_diagnostics'])} combination summaries; see the adjacent signal CSVs.",
            "",
            "## Bad replay dates",
            "",
            f"- {len(diagnostics['bad_replay_dates'])} date-horizon summaries; see the adjacent `*_bad_replay_dates.csv`.",
            "",
            "## 阅读提醒",
            "",
            "本报告只用于解释历史缓存回放中的评分与分层问题，不修改策略，不构成投资建议。",
            "后续若要修改权重、标签计分或 focused 规则，必须在独立版本中重新设计、测试和回放。",
        ]
    )
    return chr(10).join(lines)



def _infer_report_date(source_path: Path, evaluated: pd.DataFrame) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", source_path.name)
    if match:
        return match.group(1)
    dates = pd.to_datetime(evaluated["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot infer scoring diagnostics report date")
    return dates.max().date().isoformat()


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"
