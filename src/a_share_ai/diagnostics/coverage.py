from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

LIMITED_SMOKE_NOTICE = "LIMITED OPERATIONAL SMOKE / NOT A RECOMMENDATION"
PREVIEW_REASON_LIMIT = 160


@dataclass(frozen=True)
class CoverageSummary:
    report_date: date
    requested_limit: int
    cache_dir: Path
    manifest_path: Path
    manifest_exists: bool
    total_symbols: int
    ok_symbols: int
    failed_symbols: int
    success_rate: float
    status_counts: dict[str, int]
    failure_buckets: dict[str, int]
    latest_cached_min: date | None
    latest_cached_max: date | None
    preview_rows: list[dict[str, Any]]
    blocked: bool
    blocking_reason: str
    smoke_status: str = "unknown"
    smoke_error: str = ""
    funnel_counts: dict[str, int] = field(default_factory=dict)


def bucket_failure_reason(reason: str) -> str:
    text = str(reason or "").lower()
    if any(
        token in text
        for token in ("schema", "amount", "column", "nan", "malformed", "ohlcv")
    ):
        return "schema_or_amount"
    if any(
        token in text
        for token in (
            "proxy",
            "timeout",
            "remote disconnected",
            "remotedisconnected",
            "connection",
            "dns",
            "reset",
        )
    ):
        return "network_or_proxy"
    if any(
        token in text
        for token in (
            "empty history",
            "empty price history",
            "empty remote price history",
            "empty remote response",
            "empty frame",
            "empty dataframe",
            "empty response",
        )
    ):
        return "empty_remote"
    if "stale" in text or "does not cover" in text:
        return "stale_cache"
    return "unknown"


def summarize_manifest(
    manifest: pd.DataFrame | None,
    *,
    report_date: date,
    requested_limit: int,
    cache_dir: str | Path,
    manifest_path: str | Path,
    smoke_status: str = "unknown",
    smoke_error: str = "",
    funnel_counts: dict[str, int] | None = None,
) -> CoverageSummary:
    cache_dir_path = Path(cache_dir)
    manifest_path_obj = Path(manifest_path)
    normalized_funnel_counts = dict(funnel_counts or {})
    if manifest is None:
        return CoverageSummary(
            report_date=report_date,
            requested_limit=requested_limit,
            cache_dir=cache_dir_path,
            manifest_path=manifest_path_obj,
            manifest_exists=False,
            total_symbols=0,
            ok_symbols=0,
            failed_symbols=0,
            success_rate=0.0,
            status_counts={},
            failure_buckets={},
            latest_cached_min=None,
            latest_cached_max=None,
            preview_rows=[],
            blocked=True,
            blocking_reason="manifest_missing",
            smoke_status=smoke_status,
            smoke_error=smoke_error,
            funnel_counts=normalized_funnel_counts,
        )

    if manifest.empty:
        return CoverageSummary(
            report_date=report_date,
            requested_limit=requested_limit,
            cache_dir=cache_dir_path,
            manifest_path=manifest_path_obj,
            manifest_exists=True,
            total_symbols=0,
            ok_symbols=0,
            failed_symbols=0,
            success_rate=0.0,
            status_counts={},
            failure_buckets={},
            latest_cached_min=None,
            latest_cached_max=None,
            preview_rows=[],
            blocked=True,
            blocking_reason="manifest_empty",
            smoke_status=smoke_status,
            smoke_error=smoke_error,
            funnel_counts=normalized_funnel_counts,
        )

    frame = manifest.copy()
    for column in (
        "code",
        "latest_cached_date",
        "status",
        "failure_reason",
        "consecutive_failures",
    ):
        if column not in frame.columns:
            frame[column] = ""
    status = frame["status"].fillna("unknown").astype(str)
    status_counts = {
        str(key): int(value)
        for key, value in status.value_counts().sort_index().to_dict().items()
    }
    ok_symbols = int(status.eq("ok").sum())
    failed_symbols = int(status.eq("failed").sum())
    total_symbols = len(frame)
    failed_reasons = (
        frame.loc[status.eq("failed"), "failure_reason"].fillna("").astype(str)
    )
    buckets = failed_reasons.map(bucket_failure_reason)
    failure_buckets = {
        str(key): int(value)
        for key, value in buckets.value_counts().sort_index().to_dict().items()
    }
    dates = pd.to_datetime(
        frame["latest_cached_date"], errors="coerce"
    ).dt.date.dropna()

    return CoverageSummary(
        report_date=report_date,
        requested_limit=requested_limit,
        cache_dir=cache_dir_path,
        manifest_path=manifest_path_obj,
        manifest_exists=True,
        total_symbols=total_symbols,
        ok_symbols=ok_symbols,
        failed_symbols=failed_symbols,
        success_rate=ok_symbols / total_symbols if total_symbols else 0.0,
        status_counts=status_counts,
        failure_buckets=failure_buckets,
        latest_cached_min=dates.min() if not dates.empty else None,
        latest_cached_max=dates.max() if not dates.empty else None,
        preview_rows=[
            _preview_row(row) for row in frame.head(20).to_dict(orient="records")
        ],
        blocked=False,
        blocking_reason="",
        smoke_status=smoke_status,
        smoke_error=smoke_error,
        funnel_counts=normalized_funnel_counts,
    )


def _preview_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_reason = _safe_text(row.get("failure_reason", ""))
    failure_bucket = bucket_failure_reason(raw_reason) if raw_reason else ""
    reason = (
        raw_reason.replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("|", "/")
    )
    if len(reason) > PREVIEW_REASON_LIMIT:
        reason = reason[: PREVIEW_REASON_LIMIT - 1] + "…"
    return {
        "code": _safe_text(row.get("code", "")),
        "status": _safe_text(row.get("status", "")),
        "latest_cached_date": _safe_value(row.get("latest_cached_date", ""), ""),
        "failure_bucket": failure_bucket,
        "failure_reason": reason,
        "consecutive_failures": _safe_value(row.get("consecutive_failures", 0), 0),
    }


def _safe_text(value: Any) -> str:
    return str(_safe_value(value, ""))


def _safe_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    return value


def render_coverage_markdown(
    summary: CoverageSummary,
    *,
    generated_at: datetime | None = None,
) -> str:
    generated = generated_at or datetime.now(UTC).replace(microsecond=0)
    lines = [
        f"# Real Public Data Coverage Smoke - {summary.report_date.isoformat()}",
        "",
        f"> {LIMITED_SMOKE_NOTICE}",
        "",
        "> This report is an operational data-ingestion diagnostic, not an investment signal.",
        "",
        "## Run summary",
        "",
        f"- Generated at: {generated.isoformat()}",
        f"- Report date: {summary.report_date.isoformat()}",
        f"- Requested symbol limit: {summary.requested_limit}",
        f"- Cache directory: `{summary.cache_dir}`",
        f"- Manifest path: `{summary.manifest_path}`",
        f"- Manifest exists: {summary.manifest_exists}",
        f"- Smoke status: {summary.smoke_status}",
        f"- Smoke error: {summary.smoke_error or 'none'}",
        f"- Blocked: {summary.blocked}",
        f"- Blocking reason: {summary.blocking_reason or 'none'}",
        f"- Total manifest rows: {summary.total_symbols}",
        f"- OK symbols: {summary.ok_symbols}",
        f"- Failed symbols: {summary.failed_symbols}",
        f"- Success rate: {summary.success_rate:.2%}",
        f"- Latest cached date min: {_format_optional_date(summary.latest_cached_min)}",
        f"- Latest cached date max: {_format_optional_date(summary.latest_cached_max)}",
        "",
        "## Universe funnel",
        "",
        _render_funnel_table(summary),
        "",
        "## Manifest status counts",
        "",
        _render_count_table(summary.status_counts, "status"),
        "",
        "## Failure reason buckets",
        "",
        _render_count_table(summary.failure_buckets, "bucket"),
        "",
        "## Per-symbol manifest preview",
        "",
        _render_preview_table(summary.preview_rows),
        "",
        "## Operational interpretation",
        "",
        _interpret_summary(summary),
        "",
    ]
    return "\n".join(lines)


def render_coverage_terminal_summary(summary: CoverageSummary) -> str:
    latest_range = (
        f"{_format_optional_date(summary.latest_cached_min)} to "
        f"{_format_optional_date(summary.latest_cached_max)}"
    )
    failure_buckets = (
        ", ".join(
            f"{key}={value}" for key, value in sorted(summary.failure_buckets.items())
        )
        or "none"
    )
    lines = [
        "Coverage summary:",
        f"  Smoke status: {summary.smoke_status}",
        f"  Requested limit: {summary.requested_limit}",
        f"  Report date: {summary.report_date.isoformat()}",
        f"  Manifest OK/failed: {summary.ok_symbols}/{summary.failed_symbols}",
        f"  Success rate: {summary.success_rate:.2%}",
        f"  Latest cached date range: {latest_range}",
        f"  Failure buckets: {failure_buckets}",
    ]
    if summary.smoke_error:
        lines.append(f"  Smoke error: {summary.smoke_error}")
    return "\n".join(lines)


def write_coverage_report(
    summary: CoverageSummary,
    output_dir: str | Path,
    *,
    generated_at: datetime | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"{summary.report_date.isoformat()}_coverage_report.md"
    report_path.write_text(
        render_coverage_markdown(summary, generated_at=generated_at),
        encoding="utf-8",
    )
    return report_path


def _format_optional_date(value: date | None) -> str:
    return value.isoformat() if value is not None else "none"


def _render_funnel_table(summary: CoverageSummary) -> str:
    counts = {
        "requested_symbols": summary.requested_limit,
        "pre_fetch_excluded": summary.funnel_counts.get("pre_fetch_excluded", 0),
        "fetch_attempted": summary.funnel_counts.get(
            "fetch_attempted", summary.total_symbols
        ),
        "cached_ok": summary.funnel_counts.get("cached_ok", summary.ok_symbols),
        "post_fetch_excluded": summary.funnel_counts.get("post_fetch_excluded", 0),
        "final_candidates": summary.funnel_counts.get("final_candidates", 0),
    }
    rows = ["| stage | count |", "| --- | ---: |"]
    rows.extend(f"| {key} | {value} |" for key, value in counts.items())
    return "\n".join(rows)


def _render_count_table(counts: dict[str, int], label: str) -> str:
    if not counts:
        return f"| {label} | count |\n| --- | ---: |\n| none | 0 |"
    rows = [f"| {label} | count |", "| --- | ---:"]
    rows.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    return "\n".join(rows)


def _render_preview_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "| code | status | latest_cached_date | failure_bucket | consecutive_failures | failure_reason |\n| --- | --- | --- | --- | ---: | --- |\n| none | none | none | none | 0 | none |"
    output = [
        "| code | status | latest_cached_date | failure_bucket | consecutive_failures | failure_reason |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        output.append(
            "| {code} | {status} | {latest_cached_date} | {failure_bucket} | {consecutive_failures} | {failure_reason} |".format(
                code=row.get("code", ""),
                status=row.get("status", ""),
                latest_cached_date=row.get("latest_cached_date", ""),
                failure_bucket=row.get("failure_bucket", ""),
                consecutive_failures=row.get("consecutive_failures", 0),
                failure_reason=str(row.get("failure_reason", "")).replace("|", "/"),
            )
        )
    return "\n".join(output)


def _interpret_summary(summary: CoverageSummary) -> str:
    if summary.smoke_status == "provider_failed":
        return (
            "The current smoke status is `provider_failed`. Manifest coverage below "
            "describes cached observations only and does not indicate that the current "
            "smoke completed."
        )
    if summary.blocked:
        return f"Coverage smoke is blocked because `{summary.blocking_reason}`. Inspect the cache path and upstream data access before increasing the symbol limit."
    if summary.total_symbols == 0:
        return "Coverage smoke produced no manifest rows. Keep the run bounded and inspect provider startup failures."
    if summary.success_rate >= 0.8:
        return "Coverage is sufficient for the next diagnostic step. Consider trying a larger bounded run such as 50 symbols after reviewing any failed rows."
    if summary.failure_buckets:
        dominant = max(summary.failure_buckets.items(), key=lambda item: item[1])[0]
        return f"Coverage is below the preferred diagnostic threshold. The dominant failure bucket is `{dominant}`; inspect those raw failure reasons before increasing the limit."
    return "Coverage is below the preferred diagnostic threshold. Inspect manifest rows before increasing the limit."
