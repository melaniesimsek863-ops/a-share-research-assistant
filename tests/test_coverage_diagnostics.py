from a_share_ai.diagnostics.coverage import bucket_failure_reason


def test_bucket_failure_reason_classifies_stable_categories() -> None:
    assert (
        bucket_failure_reason("HTTPSConnectionPool proxy RemoteDisconnected")
        == "network_or_proxy"
    )
    assert bucket_failure_reason("RemoteDisconnected") == "network_or_proxy"
    assert bucket_failure_reason("empty remote price history") == "empty_remote"
    assert bucket_failure_reason("empty amount") == "schema_or_amount"
    assert bucket_failure_reason("empty remote metadata") == "unknown"
    assert (
        bucket_failure_reason("missing required columns: amount") == "schema_or_amount"
    )
    assert bucket_failure_reason("cached data stale for report date") == "stale_cache"
    assert bucket_failure_reason("unexpected vendor response") == "unknown"


from datetime import date
from pathlib import Path

import pandas as pd

from a_share_ai.diagnostics import coverage
from a_share_ai.diagnostics.coverage import summarize_manifest


def test_summarize_manifest_counts_statuses_and_dates(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "code": "000001",
                "latest_cached_date": date(2026, 7, 31),
                "status": "ok",
                "failure_reason": "",
                "consecutive_failures": 0,
            },
            {
                "code": "000002",
                "latest_cached_date": date(2026, 7, 30),
                "status": "failed",
                "failure_reason": "HTTPSConnectionPool proxy RemoteDisconnected",
                "consecutive_failures": 1,
            },
            {
                "code": "000006",
                "latest_cached_date": None,
                "status": "failed",
                "failure_reason": "missing required columns: amount",
                "consecutive_failures": 2,
            },
        ]
    )

    summary = summarize_manifest(
        manifest,
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )

    assert summary.total_symbols == 3
    assert summary.ok_symbols == 1
    assert summary.failed_symbols == 2
    assert summary.success_rate == 1 / 3
    assert summary.status_counts == {"failed": 2, "ok": 1}
    assert summary.failure_buckets == {"network_or_proxy": 1, "schema_or_amount": 1}
    assert summary.latest_cached_min == date(2026, 7, 30)
    assert summary.latest_cached_max == date(2026, 7, 31)
    assert summary.blocked is False
    assert summary.blocking_reason == ""


from datetime import UTC, datetime

from a_share_ai.diagnostics.coverage import (
    render_coverage_markdown,
    write_coverage_report,
)


def test_summarize_missing_and_empty_manifest_are_blocked(tmp_path: Path) -> None:
    missing = summarize_manifest(
        None,
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )
    empty = summarize_manifest(
        pd.DataFrame(),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )

    assert missing.blocked is True
    assert missing.blocking_reason == "manifest_missing"
    assert empty.blocked is True
    assert empty.blocking_reason == "manifest_empty"
    assert empty.success_rate == 0.0


def test_render_coverage_markdown_contains_notice_and_operational_summary(
    tmp_path: Path,
) -> None:
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000001",
                    "latest_cached_date": date(2026, 7, 31),
                    "status": "ok",
                    "failure_reason": "",
                    "consecutive_failures": 0,
                }
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )

    markdown = render_coverage_markdown(
        summary,
        generated_at=datetime(2026, 8, 2, 1, 30, tzinfo=UTC),
    )

    assert "# Real Public Data Coverage Smoke - 2026-07-31" in markdown
    assert "LIMITED OPERATIONAL SMOKE / NOT A RECOMMENDATION" in markdown
    assert "Success rate: 100.00%" in markdown
    assert "buy" not in markdown.lower()
    assert "sell" not in markdown.lower()


def test_render_coverage_markdown_explains_universe_funnel(tmp_path: Path) -> None:
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000001",
                    "latest_cached_date": date(2026, 7, 31),
                    "status": "ok",
                    "failure_reason": "",
                    "consecutive_failures": 0,
                }
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
        funnel_counts={
            "pre_fetch_excluded": 2,
            "fetch_attempted": 18,
            "cached_ok": 18,
            "post_fetch_excluded": 7,
            "final_candidates": 11,
        },
    )

    markdown = render_coverage_markdown(summary)

    assert "## Universe funnel" in markdown
    assert "| requested_symbols | 20 |" in markdown
    assert "| pre_fetch_excluded | 2 |" in markdown
    assert "| fetch_attempted | 18 |" in markdown
    assert "| cached_ok | 18 |" in markdown
    assert "| post_fetch_excluded | 7 |" in markdown
    assert "| final_candidates | 11 |" in markdown


def test_provider_failure_report_does_not_imply_current_smoke_completed(
    tmp_path: Path,
) -> None:
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000001",
                    "latest_cached_date": date(2026, 7, 31),
                    "status": "ok",
                    "failure_reason": "",
                    "consecutive_failures": 0,
                }
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
        smoke_status="provider_failed",
        smoke_error="offline provider failure",
    )

    markdown = render_coverage_markdown(summary)

    assert "Smoke status: provider_failed" in markdown
    assert "Smoke error: offline provider failure" in markdown
    assert "cached observations only" in markdown
    assert "Coverage is sufficient for the next diagnostic step" not in markdown


def test_terminal_summary_contains_required_coverage_fields(tmp_path: Path) -> None:
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000001",
                    "latest_cached_date": date(2026, 7, 30),
                    "status": "ok",
                    "failure_reason": "",
                    "consecutive_failures": 0,
                },
                {
                    "code": "000002",
                    "latest_cached_date": date(2026, 7, 31),
                    "status": "failed",
                    "failure_reason": "RemoteDisconnected",
                    "consecutive_failures": 1,
                },
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
        smoke_status="completed",
    )

    text = coverage.render_coverage_terminal_summary(summary)

    assert "Smoke status: completed" in text
    assert "Requested limit: 20" in text
    assert "Report date: 2026-07-31" in text
    assert "Manifest OK/failed: 1/1" in text
    assert "Success rate: 50.00%" in text
    assert "Latest cached date range: 2026-07-30 to 2026-07-31" in text
    assert "Failure buckets: network_or_proxy=1" in text


def test_write_coverage_report_creates_markdown_file(tmp_path: Path) -> None:
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000001",
                    "latest_cached_date": date(2026, 7, 31),
                    "status": "ok",
                    "failure_reason": "",
                    "consecutive_failures": 0,
                }
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )

    path = write_coverage_report(
        summary,
        tmp_path / "coverage",
        generated_at=datetime(2026, 8, 2, 1, 30, tzinfo=UTC),
    )

    assert path == tmp_path / "coverage" / "2026-07-31_coverage_report.md"
    assert path.exists()
    assert "not an investment signal" in path.read_text(encoding="utf-8")


def test_preview_reason_is_single_line_and_markdown_safe(tmp_path: Path) -> None:
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000001",
                    "latest_cached_date": date(2026, 7, 31),
                    "status": "failed",
                    "failure_reason": "line1\nline2\r\nline3|pipe",
                    "consecutive_failures": 1,
                }
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )

    markdown = render_coverage_markdown(summary)
    preview_data_rows = [
        line for line in markdown.splitlines() if line.startswith("| 000001 |")
    ]

    assert summary.preview_rows[0]["failure_reason"] == "line1 line2 line3/pipe"
    assert len(preview_data_rows) == 1
    assert "line1 line2 line3/pipe" in preview_data_rows[0]
    assert "\nline2" not in markdown
    assert "line3|pipe" not in markdown


def test_preview_bucket_uses_full_reason_before_display_truncation(
    tmp_path: Path,
) -> None:
    raw_reason = "x" * 180 + " timeout after provider request"
    summary = summarize_manifest(
        pd.DataFrame(
            [
                {
                    "code": "000002",
                    "latest_cached_date": None,
                    "status": "failed",
                    "failure_reason": raw_reason,
                    "consecutive_failures": 1,
                }
            ]
        ),
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "cache" / "manifest.csv",
    )

    assert summary.failure_buckets == {"network_or_proxy": 1}
    assert summary.preview_rows[0]["failure_bucket"] == "network_or_proxy"
    assert summary.preview_rows[0]["failure_reason"].endswith("…")
