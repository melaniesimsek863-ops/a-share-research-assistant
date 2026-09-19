import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.smoke_akshare_limited as smoke
from a_share_ai.data.cached_provider import CachedMarketDataProvider
from a_share_ai.diagnostics.coverage import summarize_manifest


class FakeAkShareProvider:
    provider_name = "akshare"


@pytest.fixture(autouse=True)
def offline_provider(monkeypatch):
    # These CLI/report tests must not depend on the optional live-data SDK.
    monkeypatch.setattr(smoke, "AkShareProvider", FakeAkShareProvider)


def test_build_limited_provider_without_cache_uses_existing_limited_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smoke, "AkShareProvider", FakeAkShareProvider)

    provider = smoke.build_limited_provider(
        limit=5,
        report_date=date(2026, 8, 1),
        cache_enabled=False,
        cache_dir="data/raw/akshare-smoke",
    )

    assert isinstance(provider, smoke.LimitedProvider)
    assert provider.provider_name == "akshare-limited-smoke"


def test_build_limited_provider_with_cache_wraps_limited_provider(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(smoke, "AkShareProvider", FakeAkShareProvider)

    provider = smoke.build_limited_provider(
        limit=5,
        report_date=date(2026, 8, 1),
        cache_enabled=True,
        cache_dir=tmp_path / "akshare-smoke",
    )

    assert isinstance(provider, CachedMarketDataProvider)
    assert provider.provider_name == "akshare-cached"
    assert isinstance(provider.inner, smoke.LimitedProvider)
    assert provider.report_date == date(2026, 8, 1)
    assert provider.cache.base_dir == tmp_path / "akshare-smoke"


def test_default_coverage_output_dir_is_limit_specific() -> None:
    assert smoke.default_coverage_output_dir(50) == (
        Path("outputs") / "smoke-akshare-limited" / "coverage-limit50"
    )


def test_resolve_coverage_output_dir_honors_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-coverage"

    assert smoke.resolve_coverage_output_dir(20, explicit) == explicit


def test_write_coverage_report_from_manifest_reads_manifest_and_writes_report(
    tmp_path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pd.DataFrame(
        [
            {
                "code": "000001",
                "latest_cached_date": "2026-07-31",
                "last_attempt_at": "2026-08-01T16:59:03+00:00",
                "last_success_at": "2026-08-01T16:59:03+00:00",
                "status": "ok",
                "failure_reason": "",
                "consecutive_failures": 0,
            }
        ]
    ).to_csv(cache_dir / "manifest.csv", index=False)

    report_path, summary = smoke.write_coverage_report_from_manifest(
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=cache_dir,
        output_dir=tmp_path / "coverage",
    )

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Success rate: 100.00%" in text
    assert "LIMITED OPERATIONAL SMOKE / NOT A RECOMMENDATION" in text
    assert summary.preview_rows[0]["code"] == "000001"
    assert summary.preview_rows[0]["failure_reason"] == ""
    assert summary.preview_rows[0]["failure_bucket"] == ""
    assert summary.failure_buckets == {}
    assert "nan" not in text.lower()


def test_write_coverage_report_from_manifest_handles_missing_manifest(tmp_path) -> None:
    report_path, summary = smoke.write_coverage_report_from_manifest(
        report_date=date(2026, 7, 31),
        requested_limit=20,
        cache_dir=tmp_path / "missing-cache",
        output_dir=tmp_path / "coverage",
    )

    text = report_path.read_text(encoding="utf-8")
    assert "Blocking reason: manifest_missing" in text
    assert "Success rate: 0.00%" in text
    assert summary.manifest_exists is False


def _coverage_summary(*, smoke_status: str, smoke_error: str = ""):
    return summarize_manifest(
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
        cache_dir="data/raw/akshare-smoke",
        manifest_path="data/raw/akshare-smoke/manifest.csv",
        smoke_status=smoke_status,
        smoke_error=smoke_error,
    )


def test_main_without_coverage_report_does_not_call_coverage_writer(
    monkeypatch, capsys
) -> None:
    def raise_provider_error(*args, **kwargs):
        raise smoke.DataProviderError("offline provider failure")

    def coverage_writer_should_not_run(*args, **kwargs):
        raise AssertionError("coverage writer must not run without --coverage-report")

    monkeypatch.setattr(sys, "argv", ["smoke", "--date", "2026-07-31"])
    monkeypatch.setattr(smoke, "run_daily_pipeline", raise_provider_error)
    monkeypatch.setattr(
        smoke, "write_coverage_report_from_manifest", coverage_writer_should_not_run
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke.main()

    assert exc_info.value.code == 2
    assert "Data provider failed: offline provider failure" in capsys.readouterr().err


def test_main_with_coverage_report_writes_default_diagnostic_path(
    monkeypatch, capsys
) -> None:
    default_coverage_path = (
        Path("outputs") / "smoke-akshare-limited" / "coverage-limit20" / "coverage.md"
    )
    result = {
        "markdown_path": Path("outputs") / "report.md",
        "excel_path": Path("outputs") / "signals.xlsx",
        "data_quality": SimpleNamespace(quality_status="ok"),
    }
    coverage_calls = []

    summary = _coverage_summary(smoke_status="completed")

    def write_coverage_report(
        *,
        report_date,
        requested_limit,
        cache_dir,
        smoke_status,
        smoke_error="",
        funnel_counts=None,
        output_dir=None,
    ):
        coverage_calls.append(
            (
                report_date,
                requested_limit,
                cache_dir,
                smoke_status,
                smoke_error,
                output_dir,
            )
        )
        return default_coverage_path, summary

    monkeypatch.setattr(
        sys, "argv", ["smoke", "--date", "2026-07-31", "--coverage-report"]
    )
    monkeypatch.setattr(smoke, "run_daily_pipeline", lambda *args, **kwargs: result)
    monkeypatch.setattr(smoke, "finalize_limited_smoke", lambda value: value)
    monkeypatch.setattr(
        smoke, "write_coverage_report_from_manifest", write_coverage_report
    )

    smoke.main()

    assert coverage_calls == [
        (
            date(2026, 7, 31),
            20,
            "data/raw/akshare-smoke",
            "completed",
            "",
            Path("outputs") / "smoke-akshare-limited" / "coverage-limit20",
        )
    ]
    output = capsys.readouterr().out
    assert "Requested limit: 20" in output
    assert "Report date: 2026-07-31" in output
    assert "Manifest OK/failed: 1/1" in output
    assert "Success rate: 50.00%" in output
    assert "Latest cached date range: 2026-07-30 to 2026-07-31" in output
    assert "Failure buckets: network_or_proxy=1" in output
    assert f"Coverage report: {default_coverage_path}" in output


def test_main_passes_pipeline_funnel_counts_to_coverage_writer(
    monkeypatch, capsys
) -> None:
    result = {
        "markdown_path": Path("outputs") / "report.md",
        "excel_path": Path("outputs") / "signals.xlsx",
        "data_quality": SimpleNamespace(quality_status="ok"),
        "candidates": pd.DataFrame([{"code": "000001"}, {"code": "000011"}]),
        "excluded": pd.DataFrame(
            [
                {"code": "000010", "exclude_reason": "st_or_delisting_risk"},
                {"code": "000008", "exclude_reason": "low_liquidity"},
            ]
        ),
    }
    coverage_calls = []
    summary = _coverage_summary(smoke_status="completed")

    def write_coverage_report(
        *,
        report_date,
        requested_limit,
        cache_dir,
        smoke_status,
        smoke_error="",
        funnel_counts=None,
        output_dir=None,
    ):
        coverage_calls.append((funnel_counts, output_dir))
        return Path("coverage.md"), summary

    monkeypatch.setattr(
        sys, "argv", ["smoke", "--date", "2026-07-31", "--coverage-report"]
    )
    monkeypatch.setattr(smoke, "run_daily_pipeline", lambda *args, **kwargs: result)
    monkeypatch.setattr(smoke, "finalize_limited_smoke", lambda value: value)
    monkeypatch.setattr(
        smoke, "write_coverage_report_from_manifest", write_coverage_report
    )

    smoke.main()

    assert coverage_calls == [
        (
            {
                "pre_fetch_excluded": 1,
                "post_fetch_excluded": 1,
                "final_candidates": 2,
            },
            Path("outputs") / "smoke-akshare-limited" / "coverage-limit20",
        )
    ]


def test_main_partial_coverage_only_writes_blocked_report_without_pipeline(
    monkeypatch, tmp_path, capsys
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pd.DataFrame(
        [
            {
                "code": "000001",
                "latest_cached_date": "2026-07-31",
                "last_attempt_at": "2026-08-02T04:58:05+00:00",
                "last_success_at": "2026-08-02T04:58:05+00:00",
                "status": "ok",
                "failure_reason": "",
                "consecutive_failures": 0,
            }
        ]
    ).to_csv(cache_dir / "manifest.csv", index=False)
    output_dir = tmp_path / "partial-coverage"

    def pipeline_should_not_run(*args, **kwargs):
        raise AssertionError("partial coverage must not run the pipeline")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke",
            "--date",
            "2026-07-31",
            "--limit",
            "50",
            "--cache-dir",
            str(cache_dir),
            "--coverage-output-dir",
            str(output_dir),
            "--partial-coverage-only",
            "--partial-status",
            "timeout_partial",
            "--partial-error",
            "command timed out after 900 seconds",
        ],
    )
    monkeypatch.setattr(smoke, "run_daily_pipeline", pipeline_should_not_run)

    smoke.main()

    report_path = output_dir / "2026-07-31_coverage_report.md"
    text = report_path.read_text(encoding="utf-8")
    assert "Smoke status: timeout_partial" in text
    assert "Smoke error: command timed out after 900 seconds" in text
    assert "Blocked: True" in text
    assert "Blocking reason: timeout_partial" in text
    assert "| requested_symbols | 50 |" in text
    assert "| fetch_attempted | 1 |" in text
    assert "| cached_ok | 1 |" in text
    assert f"Coverage report: {report_path}" in capsys.readouterr().out


def test_main_writes_coverage_report_before_provider_failure_exit(
    monkeypatch, capsys
) -> None:
    coverage_path = (
        Path("outputs") / "smoke-akshare-limited" / "coverage-limit20" / "coverage.md"
    )
    coverage_calls = []

    def raise_provider_error(*args, **kwargs):
        raise smoke.DataProviderError("offline provider failure")

    summary = _coverage_summary(
        smoke_status="provider_failed", smoke_error="offline provider failure"
    )

    def write_coverage_report(
        *,
        report_date,
        requested_limit,
        cache_dir,
        smoke_status,
        smoke_error,
        output_dir=None,
    ):
        coverage_calls.append(
            (
                report_date,
                requested_limit,
                cache_dir,
                smoke_status,
                smoke_error,
                output_dir,
            )
        )
        return coverage_path, summary

    monkeypatch.setattr(
        sys, "argv", ["smoke", "--date", "2026-07-31", "--coverage-report"]
    )
    monkeypatch.setattr(smoke, "run_daily_pipeline", raise_provider_error)
    monkeypatch.setattr(
        smoke, "write_coverage_report_from_manifest", write_coverage_report
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert coverage_calls == [
        (
            date(2026, 7, 31),
            20,
            "data/raw/akshare-smoke",
            "provider_failed",
            "offline provider failure",
            Path("outputs") / "smoke-akshare-limited" / "coverage-limit20",
        )
    ]
    assert "Smoke status: provider_failed" in captured.out
    assert f"Coverage report: {coverage_path}" in captured.out
    assert "Data provider failed: offline provider failure" in captured.err


def test_provider_error_survives_coverage_report_failure(monkeypatch, capsys) -> None:
    def raise_provider_error(*args, **kwargs):
        raise smoke.DataProviderError("offline provider failure")

    def raise_coverage_error(*args, **kwargs):
        raise OSError("manifest is locked")

    monkeypatch.setattr(
        sys, "argv", ["smoke", "--date", "2026-07-31", "--coverage-report"]
    )
    monkeypatch.setattr(smoke, "run_daily_pipeline", raise_provider_error)
    monkeypatch.setattr(
        smoke, "write_coverage_report_from_manifest", raise_coverage_error
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Coverage report failed: manifest is locked" in captured.err
    assert "Data provider failed: offline provider failure" in captured.err
