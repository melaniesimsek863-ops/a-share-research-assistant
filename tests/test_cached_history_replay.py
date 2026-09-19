import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from a_share_ai.backtest.cached_replay import (
    CachedReplayConfig,
    CachedReplayResult,
    generate_candidate_snapshots,
    load_cached_price_history,
    run_cached_history_replay,
    select_replay_dates,
    write_cached_replay_outputs,
)
from a_share_ai.data.cache import CsvMarketCache
from tests.fixtures.sample_data import sample_price_history


def _bars(code: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": trading_date,
                "code": code,
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
                "amount": 15,
            }
            for trading_date in dates
        ]
    )


def test_load_cached_price_history_uses_only_ok_manifest_rows_with_daily_files(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "cache")
    available_dates = [date(2026, 7, 29), date(2026, 7, 30)]
    cache.write_daily("000001", _bars("000001", available_dates))
    cache.write_daily("000002", _bars("000002", available_dates))
    cache.write_daily("000003", _bars("000003", available_dates))
    cache.update_manifest_success("000001", available_dates[-1])
    cache.update_manifest_failure("000002", "timeout")
    cache.update_manifest_success("000003", available_dates[-1])

    history = load_cached_price_history(tmp_path / "cache")

    assert history["code"].unique().tolist() == ["000001", "000003"]
    assert "000002" not in history["code"].unique()
    assert history["date"].tolist() == [date(2026, 7, 29), date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 30)]


def test_load_cached_price_history_skips_ok_manifest_rows_without_daily_csv(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "cache")
    cache.update_manifest_success("000001", date(2026, 7, 30))

    with pytest.raises(ValueError, match="No usable cached price history"):
        load_cached_price_history(tmp_path / "cache")


def test_load_cached_price_history_rejects_missing_cache_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Cache directory does not exist"):
        load_cached_price_history(tmp_path / "missing")


def test_select_replay_dates_returns_latest_unique_dates_up_to_end_date() -> None:
    price_history = pd.DataFrame(
        {
            "date": [
                "2026-07-30",
                "2026-07-28",
                "2026-07-30",
                "2026-07-29",
                "2026-07-31",
            ],
            "code": ["000001", "000001", "000002", "000001", "000001"],
        }
    )

    replay_dates = select_replay_dates(price_history, date(2026, 7, 30), replay_days=2)

    assert replay_dates == [date(2026, 7, 29), date(2026, 7, 30)]


def test_cached_replay_config_has_replay_defaults() -> None:
    config = CachedReplayConfig("cache-dir")

    assert config.cache_dir == "cache-dir"
    assert config.end_date is None
    assert config.replay_days == 60

def test_generate_candidate_snapshots_ignores_prices_after_replay_date() -> None:
    price_history = sample_price_history(days=80, trend=0.002)
    replay_date = price_history.iloc[69]["date"]
    altered_history = price_history.copy()
    future_rows = altered_history["date"] > replay_date
    altered_history.loc[future_rows, ["open", "high", "low", "close"]] *= 100

    baseline = generate_candidate_snapshots(
        price_history,
        [replay_date],
        "configs/default.yaml",
    )
    altered = generate_candidate_snapshots(
        altered_history,
        [replay_date],
        "configs/default.yaml",
    )

    audit_columns = [
        "date",
        "code",
        "medium_term_score",
        "timing_score",
        "total_score",
        "tier",
    ]
    pd.testing.assert_frame_equal(
        baseline[audit_columns].reset_index(drop=True),
        altered[audit_columns].reset_index(drop=True),
    )
    assert baseline["date"].eq(replay_date).all()


def test_write_cached_replay_outputs_writes_dated_auditable_research_files(tmp_path) -> None:
    # Catches: writer omitting an audit artifact, misdating its filename, or dropping research limits.
    result = CachedReplayResult(
        replay_dates=(date(2026, 7, 29), date(2026, 7, 30)),
        candidate_snapshots=pd.DataFrame(
            [{"date": date(2026, 7, 30), "code": "000001", "total_score": 88.0}]
        ),
        evaluated_candidates=pd.DataFrame(
            [{"date": date(2026, 7, 30), "code": "000001", "forward_return_1d": 0.1}]
        ),
        summary=pd.DataFrame(
            [{"horizon": 1, "sample_count": 1, "avg_return": 0.1, "median_return": 0.1, "win_rate": 1.0}]
        ),
    )

    paths = write_cached_replay_outputs(result, tmp_path / "replay-output")

    assert paths == {
        "summary": tmp_path / "replay-output" / "2026-07-30_replay_summary.md",
        "candidates": tmp_path / "replay-output" / "2026-07-30_replay_candidates.csv",
        "evaluated": tmp_path / "replay-output" / "2026-07-30_replay_evaluated.csv",
    }
    assert all(path.is_file() for path in paths.values())
    summary_text = paths["summary"].read_text(encoding="utf-8")
    assert "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e" in summary_text
    assert "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014" in summary_text
    assert "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae" in summary_text
    assert "\u4e0d\u4ee3\u8868\u5168\u90e8 A \u80a1" in summary_text
    assert pd.read_csv(paths["candidates"], dtype={"code": str})["code"].tolist() == ["000001"]
    assert pd.read_csv(paths["evaluated"])["forward_return_1d"].tolist() == [0.1]


def test_run_cached_history_replay_orchestrates_local_cache_without_network(tmp_path) -> None:
    # Catches: orchestration skipping date selection or returning outputs that cannot be audited.
    cache = CsvMarketCache(tmp_path / "cache")
    available_dates = list(pd.bdate_range("2026-01-02", periods=70).date)
    cache.write_daily("000001", _bars("000001", available_dates))
    cache.update_manifest_success("000001", available_dates[-1])

    result = run_cached_history_replay(
        CachedReplayConfig(cache_dir=tmp_path / "cache", end_date=available_dates[-2], replay_days=2)
    )

    assert result.replay_dates == tuple(available_dates[-3:-1])
    assert result.candidate_snapshots["code"].tolist() == ["000001", "000001"]
    assert result.candidate_snapshots["date"].tolist() == list(result.replay_dates)

def test_cached_history_replay_cli_runs_as_subprocess_and_reports_outputs(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "cache")
    available_dates = list(pd.bdate_range("2026-01-02", periods=70).date)
    cache.write_daily("000001", _bars("000001", available_dates))
    cache.update_manifest_success("000001", available_dates[-1])
    output_dir = tmp_path / "replay-output"
    script = Path(__file__).parents[1] / "scripts" / "replay_cached_history.py"
    runtime = Path(sys.executable)

    completed = subprocess.run(
        [
            str(runtime),
            str(script),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--config",
            "configs/default.yaml",
            "--end-date",
            "2026-04-07",
            "--replay-days",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "LIMITED CACHED REPLAY / NOT A RECOMMENDATION" in completed.stdout
    assert "replay date: 2026-04-07" in completed.stdout
    assert "candidate count: 2" in completed.stdout
    assert "evaluated count: 2" in completed.stdout
    assert (output_dir / "2026-04-07_replay_summary.md").is_file()
    assert (output_dir / "2026-04-07_replay_candidates.csv").is_file()
    assert (output_dir / "2026-04-07_replay_evaluated.csv").is_file()

def test_generate_candidate_snapshots_excludes_rows_beyond_candidate_limit(tmp_path) -> None:
    # Catches: ranking exclusions leaking into the forward-return evaluation population.
    from a_share_ai.backtest.replay import run_historical_replay

    config_path = tmp_path / "one-candidate.yaml"
    config_path.write_text("strategy:\n  focused_count: 1\n  candidate_count: 1\n", encoding="utf-8")
    replay_date = sample_price_history(days=80).iloc[-1]["date"]
    price_history = pd.concat(
        [
            sample_price_history(code="000001", days=80, trend=0.002),
            sample_price_history(code="000002", days=80, trend=0.001),
        ],
        ignore_index=True,
    )

    snapshots = generate_candidate_snapshots(price_history, [replay_date], config_path)
    evaluated, _ = run_historical_replay(snapshots, price_history)

    assert snapshots["tier"].tolist() == ["focused"]
    assert len(snapshots) == 1
    assert evaluated["code"].tolist() == snapshots["code"].tolist()


def test_write_cached_replay_outputs_adds_safety_metadata_to_each_csv(tmp_path) -> None:
    # Catches: detached candidate or evaluation CSVs missing their research-only restrictions.
    result = CachedReplayResult(
        replay_dates=(date(2026, 7, 30),),
        candidate_snapshots=pd.DataFrame([{"date": date(2026, 7, 30), "code": "000001"}]),
        evaluated_candidates=pd.DataFrame([{"date": date(2026, 7, 30), "code": "000001"}]),
        summary=pd.DataFrame(),
    )

    paths = write_cached_replay_outputs(result, tmp_path / "replay-output")

    for csv_path in (paths["candidates"], paths["evaluated"]):
        metadata = pd.read_csv(csv_path)[
            ["replay_notice", "research_notice", "recommendation_notice", "universe_notice"]
        ]
        assert metadata.to_dict("records") == [
            {
                "replay_notice": "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e",
                "research_notice": "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014",
                "recommendation_notice": "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae",
                "universe_notice": "\u672c\u5730\u6709\u9650\u7f13\u5b58\u6c60\u4e0d\u4ee3\u8868\u5168\u90e8 A \u80a1",
            }
        ]


@pytest.mark.parametrize("protected_output", ["data/raw/replay-output", "cache/replay-output"])
def test_write_cached_replay_outputs_rejects_protected_output_directories(
    tmp_path, monkeypatch, protected_output: str
) -> None:
    # Catches: replay artifacts being allowed to overwrite raw inputs or the input cache.
    monkeypatch.chdir(tmp_path)
    result = CachedReplayResult(
        replay_dates=(date(2026, 7, 30),),
        candidate_snapshots=pd.DataFrame(),
        evaluated_candidates=pd.DataFrame(),
        summary=pd.DataFrame(),
        config=CachedReplayConfig(cache_dir=tmp_path / "cache"),
    )

    with pytest.raises(ValueError, match="protected"):
        write_cached_replay_outputs(result, protected_output)


def test_write_cached_replay_outputs_reports_actual_replay_day_count_when_requested_exceeds_available(tmp_path) -> None:
    # Catches: summary implying the requested replay length instead of the available-date count.
    replay_dates = select_replay_dates(
        pd.DataFrame({"date": ["2026-07-29", "2026-07-30"], "code": ["000001", "000001"]}),
        end_date=None,
        replay_days=60,
    )
    result = CachedReplayResult(
        replay_dates=tuple(replay_dates),
        candidate_snapshots=pd.DataFrame(),
        evaluated_candidates=pd.DataFrame(),
        summary=pd.DataFrame(),
    )

    paths = write_cached_replay_outputs(result, tmp_path / "replay-output")

    assert len(replay_dates) == 2
    assert "\u5b9e\u9645\u56de\u653e\u4ea4\u6613\u65e5\u6570\uff1a2" in paths["summary"].read_text(encoding="utf-8")
