from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SAFETY_LINES = [
    "REPLAY DIAGNOSTICS / \u56de\u653e\u8bca\u65ad",
    "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014",
    "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae",
    "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e",
]

RETURN_COLUMN_PATTERN = re.compile(r"^forward_return_(\d+)d$")


def build_replay_diagnostics(evaluated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    long_returns = _to_long_returns(evaluated, expand_signal_labels=False)
    with_buckets = long_returns.assign(score_bucket=_score_buckets(long_returns["total_score"]))
    signal_returns = _to_long_returns(evaluated, expand_signal_labels=True)
    return {
        "overall": _summarize(with_buckets, []),
        "by_tier": _summarize(with_buckets, ["tier"]),
        "by_score_bucket": _summarize(with_buckets, ["score_bucket"]),
        "by_signal_label": _summarize(signal_returns, ["signal_label"]),
        "by_replay_date": _summarize(with_buckets, ["date"]),
    }


def write_replay_diagnostics_outputs(
    evaluated_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    source_path = Path(evaluated_path)
    evaluated = pd.read_csv(source_path, dtype={"code": str})
    diagnostics = build_replay_diagnostics(evaluated)
    report_date = _infer_report_date(source_path, evaluated)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_path / f"{report_date}_replay_diagnostics.md",
        "overall": output_path / f"{report_date}_diagnostics_overall.csv",
        "by_tier": output_path / f"{report_date}_diagnostics_by_tier.csv",
        "by_score_bucket": output_path / f"{report_date}_diagnostics_by_score_bucket.csv",
        "by_signal_label": output_path / f"{report_date}_diagnostics_by_signal_label.csv",
        "by_replay_date": output_path / f"{report_date}_diagnostics_by_replay_date.csv",
    }

    paths["report"].write_text(
        build_replay_diagnostics_report(diagnostics, source_path.name),
        encoding="utf-8",
    )
    for key, frame in diagnostics.items():
        frame.to_csv(paths[key], index=False)
    return paths


def build_replay_diagnostics_report(
    diagnostics: dict[str, pd.DataFrame],
    source_name: str,
) -> str:
    lines = [
        "# \u5386\u53f2\u56de\u653e\u8bca\u65ad\u62a5\u544a",
        "",
        *SAFETY_LINES,
        "",
        f"- source: {source_name}",
        "- scope: local cached replay outputs only; not a full A-share market claim.",
        "",
        "## \u603b\u4f53\u8868\u73b0",
        "",
        "| Horizon | Sample | Avg return | Median return | Win rate | Best | Worst |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics["overall"].iterrows():
        lines.append(
            "| {horizon}d | {sample_count} | {avg_return} | {median_return} | {win_rate} | {best_return} | {worst_return} |".format(
                horizon=int(row["horizon"]),
                sample_count=int(row["sample_count"]),
                avg_return=_format_percent(row["avg_return"]),
                median_return=_format_percent(row["median_return"]),
                win_rate=_format_percent(row["win_rate"]),
                best_return=_format_percent(row["best_return"]),
                worst_return=_format_percent(row["worst_return"]),
            )
        )
    lines.extend(
        [
            "",
            "## \u9605\u8bfb\u63d0\u9192",
            "",
            "\u8fd9\u4efd\u8bca\u65ad\u53ea\u7528\u6765\u5b9a\u4f4d\u7b56\u7565\u5728\u5386\u53f2\u7f13\u5b58\u6837\u672c\u4e2d\u7684\u5f31\u70b9\u548c\u53ef\u7591\u4f18\u52bf\uff0c\u4e0d\u80fd\u76f4\u63a5\u63a8\u5bfc\u672a\u6765\u6536\u76ca\uff0c\u4e5f\u4e0d\u5e94\u4f5c\u4e3a\u4e70\u5356\u4f9d\u636e\u3002",
            "\u8bf7\u540c\u65f6\u68c0\u67e5 by_tier\u3001by_score_bucket\u3001by_signal_label \u548c by_replay_date CSV\uff0c\u770b\u8d1f\u6536\u76ca\u662f\u5426\u96c6\u4e2d\u5728\u7279\u5b9a\u5206\u7ec4\u3002",
        ]
    )
    return "\n".join(lines)


def _to_long_returns(evaluated: pd.DataFrame, expand_signal_labels: bool) -> pd.DataFrame:
    records = []
    return_columns = _return_columns(evaluated)
    base_columns = ["date", "code", "tier", "total_score", "signal_labels"]
    missing_base_columns = [column for column in base_columns if column not in evaluated.columns]
    if missing_base_columns:
        raise ValueError(f"Missing evaluated columns: {missing_base_columns}")

    for _, row in evaluated.iterrows():
        signal_labels = _split_signal_labels(row["signal_labels"])
        if not signal_labels:
            signal_labels = ["unknown"]
        labels_for_row = signal_labels if expand_signal_labels else ["all"]
        for horizon, column in return_columns:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            for signal_label in labels_for_row:
                records.append(
                    {
                        "date": row["date"],
                        "code": row["code"],
                        "tier": row["tier"],
                        "total_score": row["total_score"],
                        "signal_label": signal_label,
                        "horizon": horizon,
                        "forward_return": float(value),
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=["date", "code", "tier", "total_score", "signal_label", "horizon", "forward_return"],
    )


def _return_columns(evaluated: pd.DataFrame) -> list[tuple[int, str]]:
    matches = []
    for column in evaluated.columns:
        match = RETURN_COLUMN_PATTERN.match(str(column))
        if match:
            matches.append((int(match.group(1)), str(column)))
    if not matches:
        raise ValueError("No forward_return_*d columns found")
    return sorted(matches)


def _summarize(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    output_columns = [
        *group_columns,
        "horizon",
        "sample_count",
        "avg_return",
        "median_return",
        "win_rate",
        "best_return",
        "worst_return",
    ]
    if data.empty:
        return pd.DataFrame(columns=output_columns)
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
    summary = summary.merge(win_rate, on=grouping)
    return summary[output_columns].sort_values(grouping).reset_index(drop=True)


def _score_buckets(scores: pd.Series) -> pd.Series:
    numeric_scores = pd.to_numeric(scores, errors="coerce")
    return pd.cut(
        numeric_scores,
        bins=[float("-inf"), 70, 80, 90, float("inf")],
        labels=["<70", "70-80", "80-90", ">=90"],
        right=False,
    ).astype("string").fillna("unknown")


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


def _infer_report_date(source_path: Path, evaluated: pd.DataFrame) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", source_path.name)
    if match:
        return match.group(1)
    dates = pd.to_datetime(evaluated["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot infer replay diagnostics report date")
    return dates.max().date().isoformat()


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"
