from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from a_share_ai.diagnostics.replay import (
    build_replay_diagnostics,
    write_replay_diagnostics_outputs,
)


def test_build_replay_diagnostics_summarizes_forward_returns_by_horizon() -> None:
    # Catches: diagnostics averaging the wrong forward-return columns or counting missing horizons as samples.
    evaluated = pd.DataFrame(
        [
            {
                "date": "2026-07-29",
                "code": "000001",
                "tier": "focused",
                "total_score": 91.0,
                "signal_labels": "volume_breakout",
                "forward_return_1d": 0.10,
                "forward_return_3d": 0.30,
            },
            {
                "date": "2026-07-29",
                "code": "000002",
                "tier": "candidate",
                "total_score": 72.0,
                "signal_labels": "pullback",
                "forward_return_1d": -0.05,
                "forward_return_3d": pd.NA,
            },
            {
                "date": "2026-07-30",
                "code": "000003",
                "tier": "focused",
                "total_score": 84.0,
                "signal_labels": "volume_breakout",
                "forward_return_1d": 0.02,
                "forward_return_3d": -0.03,
            },
        ]
    )

    diagnostics = build_replay_diagnostics(evaluated)

    overall = diagnostics["overall"].sort_values("horizon").reset_index(drop=True)
    assert overall.to_dict("records") == [
        {
            "horizon": 1,
            "sample_count": 3,
            "avg_return": 0.023333333333333334,
            "median_return": 0.02,
            "win_rate": 0.6666666666666666,
            "best_return": 0.1,
            "worst_return": -0.05,
        },
        {
            "horizon": 3,
            "sample_count": 2,
            "avg_return": 0.135,
            "median_return": 0.135,
            "win_rate": 0.5,
            "best_return": 0.3,
            "worst_return": -0.03,
        },
    ]


def test_write_replay_diagnostics_outputs_writes_auditable_report_and_group_csvs() -> None:
    # Catches: writer omitting safety notices or failing to preserve drill-down CSVs for review.
    case_dir = Path("outputs/test-replay-diagnostics-writer")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    evaluated_path = case_dir / "2026-07-30_replay_evaluated.csv"
    _write_sample_evaluated(evaluated_path)

    paths = write_replay_diagnostics_outputs(evaluated_path, case_dir / "diagnostics")

    expected_names = {
        "report": "2026-07-30_replay_diagnostics.md",
        "overall": "2026-07-30_diagnostics_overall.csv",
        "by_tier": "2026-07-30_diagnostics_by_tier.csv",
        "by_score_bucket": "2026-07-30_diagnostics_by_score_bucket.csv",
        "by_signal_label": "2026-07-30_diagnostics_by_signal_label.csv",
        "by_replay_date": "2026-07-30_diagnostics_by_replay_date.csv",
    }
    assert {key: path.name for key, path in paths.items()} == expected_names
    assert all(path.is_file() for path in paths.values())

    report = paths["report"].read_text(encoding="utf-8")
    assert "REPLAY DIAGNOSTICS / \u56de\u653e\u8bca\u65ad" in report
    assert "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014" in report
    assert "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae" in report
    assert "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e" in report

    overall = pd.read_csv(paths["overall"])
    assert overall[["horizon", "sample_count"]].to_dict("records") == [{"horizon": 1, "sample_count": 2}]
    by_tier = pd.read_csv(paths["by_tier"])
    assert set(by_tier["tier"]) == {"focused", "candidate"}


def test_replay_diagnostics_cli_runs_as_subprocess_and_reports_outputs() -> None:
    # Catches: diagnostics being available only as a library function instead of a runnable user workflow.
    case_dir = Path("outputs/test-replay-diagnostics-cli")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    evaluated_path = case_dir / "2026-07-30_replay_evaluated.csv"
    output_dir = case_dir / "diagnostics"
    _write_sample_evaluated(evaluated_path)
    script = Path(__file__).parents[1] / "scripts" / "replay_diagnostics.py"

    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            str(script),
            "--evaluated",
            str(evaluated_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "REPLAY DIAGNOSTICS / NOT A RECOMMENDATION" in completed.stdout
    assert "report path:" in completed.stdout
    assert (output_dir / "2026-07-30_replay_diagnostics.md").is_file()
    assert (output_dir / "2026-07-30_diagnostics_by_score_bucket.csv").is_file()


def _write_sample_evaluated(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "date": "2026-07-29",
                "code": "000001",
                "tier": "focused",
                "total_score": 91.0,
                "signal_labels": "volume_breakout",
                "forward_return_1d": 0.10,
            },
            {
                "date": "2026-07-30",
                "code": "000002",
                "tier": "candidate",
                "total_score": 72.0,
                "signal_labels": "pullback",
                "forward_return_1d": -0.05,
            },
        ]
    ).to_csv(path, index=False)


def test_build_replay_diagnostics_does_not_duplicate_overall_samples_for_multi_label_rows() -> None:
    # Catches: expanding signal labels before overall/tier/date summaries and inflating sample counts.
    evaluated = pd.DataFrame(
        [
            {
                "date": "2026-07-30",
                "code": "000001",
                "tier": "focused",
                "total_score": 91.0,
                "signal_labels": "volume_breakout,pullback",
                "forward_return_1d": 0.10,
            }
        ]
    )

    diagnostics = build_replay_diagnostics(evaluated)

    assert diagnostics["overall"][["horizon", "sample_count"]].to_dict("records") == [
        {"horizon": 1, "sample_count": 1}
    ]
    assert diagnostics["by_tier"][["tier", "horizon", "sample_count"]].to_dict("records") == [
        {"tier": "focused", "horizon": 1, "sample_count": 1}
    ]
    assert diagnostics["by_signal_label"]["sample_count"].sum() == 2
