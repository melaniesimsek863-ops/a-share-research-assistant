import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from a_share_ai.backtest.resumable_replay import (
    completed_replay_dates,
    merge_resumable_replay_parts,
    pending_replay_dates,
    write_replay_part,
)
from a_share_ai.data.cache import CsvMarketCache


def _bars(code: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": trading_date,
                "code": code,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
                "amount": 15,
            }
            for trading_date in dates
        ]
    )


def test_pending_replay_dates_skips_completed_dates_only_when_part_files_exist(tmp_path) -> None:
    # Catches: resume logic trusting manifest status without verifying the reusable part files.
    output_dir = tmp_path / "resume"
    completed_date = date(2026, 7, 29)
    missing_part_date = date(2026, 7, 30)
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "replay_date": completed_date.isoformat(),
                "status": "ok",
                "candidate_count": 1,
                "evaluated_count": 1,
                "error": "",
            },
            {
                "replay_date": missing_part_date.isoformat(),
                "status": "ok",
                "candidate_count": 1,
                "evaluated_count": 1,
                "error": "",
            },
        ]
    ).to_csv(output_dir / "manifest.csv", index=False)
    write_replay_part(
        output_dir,
        completed_date,
        pd.DataFrame([{"date": completed_date, "code": "000001", "total_score": 88.0}]),
        pd.DataFrame([{"date": completed_date, "code": "000001", "forward_return_1d": 0.1}]),
    )

    assert completed_replay_dates(output_dir) == {completed_date}
    assert pending_replay_dates(output_dir, [completed_date, missing_part_date]) == [missing_part_date]


def test_merge_resumable_replay_parts_writes_combined_outputs_with_safety_metadata(tmp_path) -> None:
    # Catches: completed replay parts being stranded instead of producing auditable merged outputs.
    output_dir = tmp_path / "resume"
    first_date = date(2026, 7, 29)
    second_date = date(2026, 7, 30)
    write_replay_part(
        output_dir,
        first_date,
        pd.DataFrame([{"date": first_date, "code": "000001", "total_score": 88.0}]),
        pd.DataFrame([{"date": first_date, "code": "000001", "forward_return_1d": 0.10}]),
    )
    write_replay_part(
        output_dir,
        second_date,
        pd.DataFrame([{"date": second_date, "code": "000002", "total_score": 76.0}]),
        pd.DataFrame([{"date": second_date, "code": "000002", "forward_return_1d": -0.05}]),
    )

    paths = merge_resumable_replay_parts(output_dir, [first_date, second_date])

    assert paths["candidates"] == output_dir / "2026-07-30_replay_candidates.csv"
    assert paths["evaluated"] == output_dir / "2026-07-30_replay_evaluated.csv"
    assert paths["summary"] == output_dir / "2026-07-30_replay_summary.md"
    evaluated = pd.read_csv(paths["evaluated"], dtype={"code": str})
    assert evaluated["code"].tolist() == ["000001", "000002"]
    assert evaluated["replay_notice"].unique().tolist() == [
        "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e"
    ]
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014" in summary
    assert "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae" in summary


def test_resumable_cached_replay_cli_runs_parts_and_diagnostics(tmp_path) -> None:
    # Catches: resume support existing only as library utilities instead of a user-runnable workflow.
    cache = CsvMarketCache(tmp_path / "cache")
    available_dates = list(pd.bdate_range("2026-01-02", periods=70).date)
    cache.write_daily("000001", _bars("000001", available_dates))
    cache.update_manifest_success("000001", available_dates[-1])
    output_dir = tmp_path / "resume-output"
    diagnostics_dir = tmp_path / "diagnostics"
    script = Path(__file__).parents[1] / "scripts" / "replay_cached_history_resume.py"

    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            str(script),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--config",
            "configs/default.yaml",
            "--end-date",
            "2026-04-09",
            "--replay-days",
            "2",
            "--output-dir",
            str(output_dir),
            "--diagnostics-dir",
            str(diagnostics_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "RESUMABLE CACHED REPLAY / NOT A RECOMMENDATION" in completed.stdout
    assert "pending replay dates: 2" in completed.stdout
    assert (output_dir / "manifest.csv").is_file()
    assert (output_dir / "parts" / "2026-04-08_replay_evaluated.csv").is_file()
    assert (output_dir / "parts" / "2026-04-09_replay_evaluated.csv").is_file()
    assert (output_dir / "2026-04-09_replay_evaluated.csv").is_file()
    assert (diagnostics_dir / "2026-04-09_replay_diagnostics.md").is_file()

    second_run = subprocess.run(
        [
            str(Path(sys.executable)),
            str(script),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--config",
            "configs/default.yaml",
            "--end-date",
            "2026-04-09",
            "--replay-days",
            "2",
            "--output-dir",
            str(output_dir),
            "--diagnostics-dir",
            str(diagnostics_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert second_run.returncode == 0, second_run.stderr
    assert "pending replay dates: 0" in second_run.stdout
