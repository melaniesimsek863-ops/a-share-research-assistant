from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.backtest.cached_replay import (
    CachedReplayConfig,
    CachedReplayResult,
    generate_candidate_snapshots,
    load_cached_price_history,
    select_replay_dates,
    write_cached_replay_outputs,
)
from a_share_ai.backtest.replay import ReplayConfig, run_historical_replay

DEFAULT_CACHE_DIR = "data/raw/akshare-smoke-v051-20260731-limit50"
DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_OUTPUT_DIR = "outputs/replay-cached-history"
NOTICE = "LIMITED CACHED REPLAY / NOT A RECOMMENDATION"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cached historical A-share replay")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--replay-days", type=int, default=60)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.replay_days <= 0:
        parser.error("--replay-days must be positive")

    price_history = load_cached_price_history(args.cache_dir)
    replay_dates = select_replay_dates(price_history, args.end_date, args.replay_days)
    candidates = generate_candidate_snapshots(price_history, replay_dates, args.config)
    evaluated, summary = run_historical_replay(candidates, price_history, ReplayConfig())
    result = CachedReplayResult(
        replay_dates=tuple(replay_dates),
        candidate_snapshots=candidates,
        evaluated_candidates=evaluated,
        summary=summary,
        config=CachedReplayConfig(
            cache_dir=args.cache_dir,
            end_date=args.end_date,
            replay_days=args.replay_days,
        ),
    )
    paths = write_cached_replay_outputs(result, args.output_dir)
    replay_date = replay_dates[-1].isoformat()
    print(NOTICE)
    print(f"replay date: {replay_date}")
    print(f"candidate count: {len(candidates)}")
    print(f"evaluated count: {len(evaluated)}")
    print(f"summary path: {paths['summary']}")
    print(f"candidates path: {paths['candidates']}")
    print(f"evaluated path: {paths['evaluated']}")


if __name__ == "__main__":
    main()
