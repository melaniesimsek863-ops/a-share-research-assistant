from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.backtest.resumable_replay import (
    ResumableCachedReplayConfig,
    run_resumable_cached_history_replay,
)
from a_share_ai.diagnostics.replay import write_replay_diagnostics_outputs

DEFAULT_CACHE_DIR = "data/raw/akshare-smoke-v051-20260731-limit50"
DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_OUTPUT_DIR = "outputs/replay-cached-history-resume"
NOTICE = "RESUMABLE CACHED REPLAY / NOT A RECOMMENDATION"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable cached historical A-share replay")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--replay-days", type=int, default=60)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--diagnostics-dir")
    args = parser.parse_args()
    if args.replay_days <= 0:
        parser.error("--replay-days must be positive")

    result = run_resumable_cached_history_replay(
        ResumableCachedReplayConfig(
            cache_dir=args.cache_dir,
            config_path=args.config,
            end_date=args.end_date,
            replay_days=args.replay_days,
            output_dir=args.output_dir,
        )
    )

    print(NOTICE)
    print(f"replay dates: {len(result.replay_dates)}")
    print(f"pending replay dates: {len(result.pending_dates)}")
    print(f"summary path: {result.paths['summary']}")
    print(f"candidates path: {result.paths['candidates']}")
    print(f"evaluated path: {result.paths['evaluated']}")

    if args.diagnostics_dir:
        diagnostics_paths = write_replay_diagnostics_outputs(result.paths["evaluated"], args.diagnostics_dir)
        print(f"diagnostics report path: {diagnostics_paths['report']}")


if __name__ == "__main__":
    main()
