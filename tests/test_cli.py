from __future__ import annotations

import sys
from datetime import date

from a_share_ai import cli
from a_share_ai.data.providers import DataProviderError
from a_share_ai.pipeline import run_daily_pipeline


def test_daily_cli_passes_supplied_risk_state_and_emits_observe_only_output(
    monkeypatch, tmp_path
):
    """Removing CLI risk arguments must not silently restore demo risk controls."""
    captured = {}

    def run_pipeline_with_temp_output(
        config_path,
        report_date,
        provider=None,
        provider_name=None,
        output_dir=None,
        risk_state=None,
    ):
        captured["risk_state"] = risk_state
        return run_daily_pipeline(
            config_path,
            report_date,
            provider=provider,
            provider_name=provider_name,
            output_dir=tmp_path,
            risk_state=risk_state,
        )

    monkeypatch.setattr(cli, "run_daily_pipeline", run_pipeline_with_temp_output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "a-share-ai",
            "daily",
            "--config",
            "configs/default.yaml",
            "--date",
            "2026-07-29",
            "--current-drawdown",
            "-0.16",
            "--current-holdings",
            "2",
            "--daily-new-buys",
            "1",
        ],
    )

    cli.main()

    assert captured["risk_state"].current_drawdown == -0.16
    assert captured["risk_state"].current_holdings == 2
    assert captured["risk_state"].daily_new_buys == 1
    report = (tmp_path / "reports" / f"{date(2026, 7, 29)}_daily_report.md").read_text(
        encoding="utf-8"
    )
    assert "-0.16" in report
    assert "caller-supplied" in report
    assert "observe_only" in report
    assert "0%" in report


def test_daily_cli_passes_provider_override(monkeypatch, capsys):
    """Removing --provider would prevent explicit public-data selection."""
    captured = {}

    def fake_pipeline(
        config_path,
        report_date,
        provider=None,
        provider_name=None,
        output_dir=None,
        risk_state=None,
    ):
        captured["provider_name"] = provider_name
        return {"markdown_path": "report.md", "excel_path": "signals.xlsx"}

    monkeypatch.setattr(cli, "run_daily_pipeline", fake_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "a-share-ai",
            "daily",
            "--config",
            "configs/default.yaml",
            "--date",
            "2026-07-29",
            "--provider",
            "akshare",
        ],
    )

    cli.main()

    assert captured["provider_name"] == "akshare"

def test_daily_cli_reports_provider_failure(monkeypatch, capsys):
    def fake_pipeline(*args, **kwargs):
        raise DataProviderError("akshare failed")

    monkeypatch.setattr(cli, "run_daily_pipeline", fake_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        ["a-share-ai", "daily", "--config", "configs/default.yaml", "--date", "2026-07-29", "--provider", "akshare"],
    )

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    captured = capsys.readouterr()
    assert "akshare failed" in captured.err
