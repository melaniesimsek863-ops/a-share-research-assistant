from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from a_share_ai.backtest.replay import ReplayConfig, run_historical_replay
from a_share_ai.config import load_config
from a_share_ai.data.cache import CsvMarketCache
from a_share_ai.features.technical import add_technical_features
from a_share_ai.reports.replay_report import build_replay_summary
from a_share_ai.strategies.medium_term import score_medium_term
from a_share_ai.strategies.ranking import rank_candidates
from a_share_ai.strategies.timing import score_timing


@dataclass(frozen=True)
class CachedReplayConfig:
    cache_dir: str | Path
    end_date: date | None = None
    replay_days: int = 60


@dataclass(frozen=True)
class CachedReplayResult:
    replay_dates: tuple[date, ...]
    candidate_snapshots: pd.DataFrame
    evaluated_candidates: pd.DataFrame
    summary: pd.DataFrame
    config: CachedReplayConfig | None = None


def load_cached_price_history(cache_dir: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_path}")

    cache = CsvMarketCache(cache_path)
    manifest = cache.read_manifest()
    histories: list[pd.DataFrame] = []
    for code in manifest.loc[manifest["status"].eq("ok"), "code"]:
        daily = cache.read_daily(str(code))
        if not daily.empty:
            histories.append(daily)

    if not histories:
        raise ValueError("No usable cached price history")

    return (
        pd.concat(histories, ignore_index=True)
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )


def select_replay_dates(
    price_history: pd.DataFrame,
    end_date: date | None,
    replay_days: int,
) -> list[date]:
    dates = pd.to_datetime(price_history["date"], errors="coerce").dropna().dt.date
    if end_date is not None:
        dates = dates[dates <= end_date]
    unique_dates = sorted(set(dates))
    return unique_dates[-replay_days:] if replay_days > 0 else []


def generate_candidate_snapshots(
    price_history: pd.DataFrame,
    replay_dates: list[date],
    config_path: str | Path,
) -> pd.DataFrame:
    config = load_config(config_path)
    snapshots: list[pd.DataFrame] = []
    required_feature_columns = [
        "ma20",
        "ma60",
        "return_20d",
        "return_60d",
        "volume_ratio_5_20",
        "drawdown_60d",
    ]

    for replay_date in replay_dates:
        price_history_slice = price_history.loc[
            pd.to_datetime(price_history["date"]) <= pd.Timestamp(replay_date)
        ]
        enriched = add_technical_features(price_history_slice)
        latest_rows = (
            enriched.dropna(subset=required_feature_columns)
            .sort_values(["code", "date"])
            .groupby("code", as_index=False)
            .tail(1)
        )
        scored_rows = score_timing(score_medium_term(latest_rows))
        candidates = rank_candidates(scored_rows, config.strategy)
        candidates = candidates.loc[candidates["tier"].isin(["focused", "candidate"])].copy()
        candidates["date"] = replay_date
        snapshots.append(candidates)

    if not snapshots:
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "medium_term_score",
                "timing_score",
                "total_score",
                "tier",
            ]
        )
    return pd.concat(snapshots, ignore_index=True)


def run_cached_history_replay(config: CachedReplayConfig) -> CachedReplayResult:
    price_history = load_cached_price_history(config.cache_dir)
    replay_dates = select_replay_dates(price_history, config.end_date, config.replay_days)
    candidate_snapshots = generate_candidate_snapshots(price_history, replay_dates, "configs/default.yaml")
    evaluated_candidates, summary = run_historical_replay(candidate_snapshots, price_history, ReplayConfig())
    return CachedReplayResult(
        replay_dates=tuple(replay_dates),
        candidate_snapshots=candidate_snapshots,
        evaluated_candidates=evaluated_candidates,
        summary=summary,
        config=config,
    )


def write_cached_replay_outputs(
    result: CachedReplayResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    if not result.replay_dates:
        raise ValueError("Cannot write cached replay outputs without replay dates")

    output_path = Path(output_dir).resolve()
    protected_directories = [Path("data/raw").resolve()]
    if result.config is not None:
        protected_directories.append(Path(result.config.cache_dir).resolve())
    for protected_directory in protected_directories:
        try:
            output_path.relative_to(protected_directory)
        except ValueError:
            continue
        raise ValueError(f"Cached replay output directory is protected: {protected_directory}")

    output_path.mkdir(parents=True, exist_ok=True)
    report_date = result.replay_dates[-1].isoformat()
    paths = {
        "summary": output_path / f"{report_date}_replay_summary.md",
        "candidates": output_path / f"{report_date}_replay_candidates.csv",
        "evaluated": output_path / f"{report_date}_replay_evaluated.csv",
    }
    summary_text = "\n".join(
        [
            "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e",
            "\u672c\u5730\u6709\u9650\u7f13\u5b58\u6c60\u4e0d\u4ee3\u8868\u5168\u90e8 A \u80a1\uff0c\u56de\u653e\u7ed3\u679c\u4e0d\u80fd\u4ee3\u8868\u5168\u5e02\u573a\u3002",
            "",
            build_replay_summary(result.summary, len(result.candidate_snapshots), result.replay_dates),
        ]
    )
    csv_metadata = {
        "replay_notice": "LIMITED CACHED REPLAY / \u6709\u9650\u7f13\u5b58\u6c60\u56de\u653e",
        "research_notice": "HISTORICAL RESEARCH ONLY / \u5386\u53f2\u7814\u7a76\u7528\u9014",
        "recommendation_notice": "NOT A RECOMMENDATION / \u4e0d\u6784\u6210\u6295\u8d44\u5efa\u8bae",
        "universe_notice": "\u672c\u5730\u6709\u9650\u7f13\u5b58\u6c60\u4e0d\u4ee3\u8868\u5168\u90e8 A \u80a1",
    }
    paths["summary"].write_text(summary_text, encoding="utf-8")
    result.candidate_snapshots.assign(**csv_metadata).to_csv(paths["candidates"], index=False)
    result.evaluated_candidates.assign(**csv_metadata).to_csv(paths["evaluated"], index=False)
    return paths
