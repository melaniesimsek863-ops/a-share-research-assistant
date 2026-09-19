from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from a_share_ai.backtest.cached_replay import (
    CachedReplayResult,
    generate_candidate_snapshots,
    load_cached_price_history,
    select_replay_dates,
    write_cached_replay_outputs,
)
from a_share_ai.backtest.replay import ReplayConfig, run_historical_replay

CSV_METADATA = {
    "replay_notice": "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e",
    "research_notice": "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014",
    "recommendation_notice": "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae",
    "universe_notice": "\u672c\u5730\u6709\u9650\u7f13\u5b58\u6c60\u4e0d\u4ee3\u8868\u5168\u90e8 A \u80a1",
}


@dataclass(frozen=True)
class ResumableCachedReplayConfig:
    cache_dir: str | Path
    config_path: str | Path = "configs/default.yaml"
    end_date: date | None = None
    replay_days: int = 60
    output_dir: str | Path = "outputs/replay-cached-history-resume"


@dataclass(frozen=True)
class ResumableCachedReplayResult:
    replay_dates: tuple[date, ...]
    pending_dates: tuple[date, ...]
    paths: dict[str, Path]
    config: ResumableCachedReplayConfig


def completed_replay_dates(output_dir: str | Path) -> set[date]:
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.csv"
    if not manifest_path.is_file():
        return set()
    manifest = pd.read_csv(manifest_path, dtype={"replay_date": str, "status": str})
    completed: set[date] = set()
    for _, row in manifest.iterrows():
        if row.get("status") != "ok":
            continue
        replay_date = date.fromisoformat(str(row["replay_date"]))
        paths = replay_part_paths(output_path, replay_date)
        if paths["candidates"].is_file() and paths["evaluated"].is_file():
            completed.add(replay_date)
    return completed


def pending_replay_dates(output_dir: str | Path, replay_dates: list[date]) -> list[date]:
    completed = completed_replay_dates(output_dir)
    return [replay_date for replay_date in replay_dates if replay_date not in completed]


def write_replay_part(
    output_dir: str | Path,
    replay_date: date,
    candidates: pd.DataFrame,
    evaluated: pd.DataFrame,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    paths = replay_part_paths(output_path, replay_date)
    paths["parts_dir"].mkdir(parents=True, exist_ok=True)
    candidates.assign(**CSV_METADATA).to_csv(paths["candidates"], index=False)
    evaluated.assign(**CSV_METADATA).to_csv(paths["evaluated"], index=False)
    _upsert_manifest(
        output_path,
        {
            "replay_date": replay_date.isoformat(),
            "status": "ok",
            "candidate_count": len(candidates),
            "evaluated_count": len(evaluated),
            "error": "",
        },
    )
    return paths


def merge_resumable_replay_parts(output_dir: str | Path, replay_dates: list[date]) -> dict[str, Path]:
    if not replay_dates:
        raise ValueError("Cannot merge resumable replay parts without replay dates")
    output_path = Path(output_dir)
    candidates = _read_part_frames(output_path, replay_dates, "candidates")
    evaluated = _read_part_frames(output_path, replay_dates, "evaluated")
    summary = _build_forward_summary(evaluated)
    result = CachedReplayResult(
        replay_dates=tuple(replay_dates),
        candidate_snapshots=candidates,
        evaluated_candidates=evaluated,
        summary=summary,
    )
    return write_cached_replay_outputs(result, output_path)


def run_resumable_cached_history_replay(config: ResumableCachedReplayConfig) -> ResumableCachedReplayResult:
    if config.replay_days <= 0:
        raise ValueError("replay_days must be positive")
    price_history = load_cached_price_history(config.cache_dir)
    replay_dates = select_replay_dates(price_history, config.end_date, config.replay_days)
    pending_dates = pending_replay_dates(config.output_dir, replay_dates)
    for replay_date in pending_dates:
        candidates = generate_candidate_snapshots(price_history, [replay_date], config.config_path)
        evaluated, _ = run_historical_replay(candidates, price_history, ReplayConfig())
        write_replay_part(config.output_dir, replay_date, candidates, evaluated)
    paths = merge_resumable_replay_parts(config.output_dir, replay_dates)
    return ResumableCachedReplayResult(
        replay_dates=tuple(replay_dates),
        pending_dates=tuple(pending_dates),
        paths=paths,
        config=config,
    )


def replay_part_paths(output_dir: str | Path, replay_date: date) -> dict[str, Path]:
    parts_dir = Path(output_dir) / "parts"
    prefix = replay_date.isoformat()
    return {
        "parts_dir": parts_dir,
        "candidates": parts_dir / f"{prefix}_replay_candidates.csv",
        "evaluated": parts_dir / f"{prefix}_replay_evaluated.csv",
    }


def _read_part_frames(output_dir: Path, replay_dates: list[date], kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for replay_date in replay_dates:
        paths = replay_part_paths(output_dir, replay_date)
        path = paths[kind]
        if not path.is_file():
            raise FileNotFoundError(f"Missing replay part: {path}")
        frames.append(pd.read_csv(path, dtype={"code": str}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _upsert_manifest(output_dir: Path, row: dict[str, object]) -> None:
    manifest_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, dtype={"replay_date": str})
    else:
        manifest = pd.DataFrame(columns=["replay_date", "status", "candidate_count", "evaluated_count", "error"])
    manifest = manifest.loc[manifest["replay_date"].astype(str) != str(row["replay_date"])]
    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest = manifest.sort_values("replay_date").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)


def _build_forward_summary(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, column_name in sorted(_forward_return_columns(evaluated), key=lambda item: item[0]):
        returns = pd.to_numeric(evaluated[column_name], errors="coerce").dropna()
        if returns.empty:
            continue
        rows.append(
            {
                "horizon": horizon,
                "sample_count": len(returns),
                "avg_return": returns.mean(),
                "median_return": returns.median(),
                "win_rate": (returns > 0).mean(),
            }
        )
    return pd.DataFrame(rows, columns=["horizon", "sample_count", "avg_return", "median_return", "win_rate"])


def _forward_return_columns(evaluated: pd.DataFrame) -> list[tuple[int, str]]:
    columns = []
    for column in evaluated.columns:
        text = str(column)
        if text.startswith("forward_return_") and text.endswith("d"):
            horizon_text = text.removeprefix("forward_return_").removesuffix("d")
            if horizon_text.isdigit():
                columns.append((int(horizon_text), text))
    return columns
