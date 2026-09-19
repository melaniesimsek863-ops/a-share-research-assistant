from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SAFETY_METADATA = {
    "research_notice": "HISTORICAL RESEARCH ONLY / 历史研究用途",
    "recommendation_notice": "NOT A RECOMMENDATION / 不构成投资建议",
    "scope_notice": "LIMITED CACHED REPLAY / 有限缓存池回放",
}
SAFETY_LINES = [
    "HISTORICAL RESEARCH ONLY / 历史研究用途",
    "NOT A RECOMMENDATION / 不构成投资建议",
    "LIMITED CACHED REPLAY / 有限缓存池回放",
]
STRICT_FACTOR_COLUMNS = [
    "return_20d",
    "return_60d",
    "drawdown_60d",
    "volume_ratio_5_20",
    "above_ma20",
    "ma_bullish",
    "medium_term_score",
    "timing_score",
    "total_score",
    "experimental_total_score",
]
BASE_REQUIRED_COLUMNS = [
    "date",
    "code",
    "tier",
    "signal_labels",
    "return_20d",
    "return_60d",
    "drawdown_60d",
    "volume_ratio_5_20",
    "medium_term_score",
    "timing_score",
    "total_score",
]
NUMERIC_FACTORS = [
    "return_20d",
    "return_60d",
    "drawdown_60d",
    "volume_ratio_5_20",
    "above_ma20",
    "ma_bullish",
    "medium_term_score",
    "timing_score",
    "total_score",
    "experimental_total_score",
]
RETURN_PATTERN = re.compile(r"^forward_return_(\d+)d$")


def build_factor_effectiveness_diagnostics(
    evaluated: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    _validate_columns(evaluated)
    evaluated = evaluated.reset_index(drop=True)
    returns = _return_columns(evaluated)
    factors = _factor_frame(evaluated)
    by_date = _build_by_date(evaluated, factors, returns)
    by_factor = _build_by_factor(by_date)
    return {"by_factor": by_factor, "by_date": by_date}


def build_factor_effectiveness_report(
    diagnostics: dict[str, pd.DataFrame], source_name: str
) -> str:
    by_factor = diagnostics["by_factor"]
    negative = by_factor.loc[by_factor["direction"].eq("negative"), "factor"]
    lines = [
        "# FACTOR EFFECTIVENESS / 因子有效性诊断",
        "",
        *SAFETY_LINES,
        "",
        f"- source: {source_name}",
        "- scope: historical cached replay research only; production scoring and signals are unchanged.",
        "",
        "## Factor ranking",
        "",
        "| Factor | Horizon | Mean Spearman | Q4-Q1 mean | Direction |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in by_factor.sort_values(
        ["horizon", "mean_spearman_corr", "factor"], ascending=[True, False, True]
    ).iterrows():
        lines.append(
            f"| {row['factor']} | {int(row['horizon'])}d | "
            f"{_format_number(row['mean_spearman_corr'])} | "
            f"{_format_number(row['q4_minus_q1_mean'])} | {row['direction']} |"
        )
    lines.extend(
        [
            "",
            "## Negative factor list",
            "",
            "- " + (", ".join(map(str, negative.unique())) if not negative.empty else "None identified"),
            "",
            "## Sample limitations",
            "",
            "- Results describe this limited cached replay universe and available forward-return horizons only.",
            "- Correlation and bucket comparisons are descriptive diagnostics, not causal evidence or trading instructions.",
        ]
    )
    return "\n".join(lines)


def write_factor_effectiveness_outputs(
    evaluated_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    source_path = Path(evaluated_path)
    evaluated = pd.read_csv(source_path, dtype={"code": str})
    diagnostics = build_factor_effectiveness_diagnostics(evaluated)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_date = _infer_report_date(source_path, evaluated)
    paths = {
        "report": output_path / f"{report_date}_factor_effectiveness.md",
        "by_factor": output_path / f"{report_date}_factor_effectiveness_by_factor.csv",
        "by_date": output_path / f"{report_date}_factor_effectiveness_by_date.csv",
    }
    paths["report"].write_text(
        build_factor_effectiveness_report(diagnostics, source_path.name), encoding="utf-8"
    )
    for key in ("by_factor", "by_date"):
        diagnostics[key].to_csv(paths[key], index=False)
    return paths


def _validate_columns(evaluated: pd.DataFrame) -> None:
    missing = [
        column
        for column in [*BASE_REQUIRED_COLUMNS, *STRICT_FACTOR_COLUMNS]
        if column not in evaluated.columns
    ]
    if not _return_columns(evaluated):
        missing.append("forward_return_*d")
    if missing:
        raise ValueError(f"Missing required factor effectiveness columns: {missing}")


def _return_columns(evaluated: pd.DataFrame) -> list[tuple[int, str]]:
    matches = []
    for column in evaluated.columns:
        match = RETURN_PATTERN.match(str(column))
        if match:
            matches.append((int(match.group(1)), str(column)))
    return sorted(matches)


def _factor_frame(evaluated: pd.DataFrame) -> pd.DataFrame:
    factors = evaluated.copy().reset_index(drop=True)
    factor_names = [factor for factor in NUMERIC_FACTORS if factor in factors.columns]
    label_names = sorted(
        {
            f"signal_label:{_sanitize_label(label)}"
            for value in factors["signal_labels"]
            for label in _split_labels(value)
        }
    )
    for factor in label_names:
        label = factor.removeprefix("signal_label:")
        factors[factor] = factors["signal_labels"].map(
            lambda value, expected=label: expected in {_sanitize_label(item) for item in _split_labels(value)}
        )
    return factors[["date", *factor_names, *label_names]]


def _build_by_date(
    evaluated: pd.DataFrame,
    factors: pd.DataFrame,
    returns: list[tuple[int, str]],
) -> pd.DataFrame:
    rows = []
    for factor in factors.columns[1:]:
        for date_value, date_indices in factors.groupby("date", dropna=False).groups.items():
            buckets = _date_buckets(factors.loc[date_indices, factor])
            for horizon, return_column in returns:
                values = pd.to_numeric(evaluated.loc[date_indices, return_column], errors="coerce")
                for bucket in sorted(set(buckets.dropna().astype(str))):
                    selected = values.loc[buckets.eq(bucket)].dropna()
                    rows.append(
                        _summary_row(date_value, horizon, factor, bucket, selected, factors.loc[date_indices, factor], values)
                    )
    columns = [
        "date", "horizon", "factor", "bucket", "sample_count", "avg_return",
        "median_return", "win_rate", "spearman_corr", *SAFETY_METADATA,
    ]
    return pd.DataFrame(rows, columns=columns)


def _summary_row(date_value, horizon, factor, bucket, selected, factor_values, returns):
    numeric_factor = pd.to_numeric(factor_values, errors="coerce")
    valid = numeric_factor.notna() & returns.notna()
    correlation = _spearman_corr(numeric_factor[valid], returns[valid])
    return {
        "date": date_value,
        "horizon": horizon,
        "factor": factor,
        "bucket": bucket,
        "sample_count": int(selected.count()),
        "avg_return": selected.mean(),
        "median_return": selected.median(),
        "win_rate": (selected > 0).mean() if not selected.empty else float("nan"),
        "spearman_corr": correlation,
        **SAFETY_METADATA,
    }


def _build_by_factor(by_date: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, horizon), group in by_date.groupby(["factor", "horizon"], sort=True):
        bucket_returns = group.pivot(index="date", columns="bucket", values="avg_return")
        paired = bucket_returns.reindex(columns=["q1", "q4"]).dropna()
        q4_minus_q1 = paired["q4"] - paired["q1"]
        mean_corr = group["spearman_corr"].mean()
        mean_delta = q4_minus_q1.mean()
        direction = (
            "positive" if mean_corr > 0 and mean_delta > 0
            else "negative" if mean_corr < 0 and mean_delta < 0
            else "mixed"
        )
        rows.append(
            {
                "factor": factor,
                "horizon": horizon,
                "date_count": int(group["date"].nunique()),
                "mean_spearman_corr": mean_corr,
                "spearman_corr": mean_corr,
                "median_spearman_corr": group["spearman_corr"].median(),
                "q4_minus_q1_mean": mean_delta,
                "q4_minus_q1_positive_rate": (q4_minus_q1 > 0).mean(),
                "direction": direction,
                **SAFETY_METADATA,
            }
        )
    columns = [
        "factor", "horizon", "date_count", "mean_spearman_corr",
        "median_spearman_corr", "q4_minus_q1_mean",
        "q4_minus_q1_positive_rate", "direction", *SAFETY_METADATA,
    ]
    return pd.DataFrame(rows, columns=columns)


def _date_buckets(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unique_values = numeric.dropna().drop_duplicates().sort_values()
    count = min(4, len(unique_values))
    if count == 0:
        return pd.Series(pd.NA, index=values.index, dtype="string")
    if count == 1:
        mapped = pd.Series("q1", index=unique_values.index, dtype="string")
    else:
        labels = {
            2: ["q1", "q4"],
            3: ["q1", "q3", "q4"],
            4: ["q1", "q2", "q3", "q4"],
        }[count]
        mapped = pd.Series(
            pd.qcut(unique_values, q=count, labels=labels).astype("string").to_numpy(),
            index=unique_values.index,
        )
    return numeric.map(dict(zip(unique_values, mapped))).astype("string")

def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _split_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].replace("'", "").replace('"', "")
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def _sanitize_label(value: object) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip()).strip("_").lower()
    return sanitized or "unknown"


def _infer_report_date(source_path: Path, evaluated: pd.DataFrame) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", source_path.name)
    if match:
        return match.group(1)
    dates = pd.to_datetime(evaluated["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot infer factor effectiveness report date")
    return dates.max().date().isoformat()


def _format_number(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.4f}"
