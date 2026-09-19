from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.diagnostics.replay import write_replay_diagnostics_outputs

DEFAULT_EVALUATED = "outputs/replay-cached-history/2026-07-31_replay_evaluated.csv"
DEFAULT_OUTPUT_DIR = "outputs/replay-diagnostics"
NOTICE = "REPLAY DIAGNOSTICS / NOT A RECOMMENDATION"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diagnostics from a cached historical replay evaluated CSV")
    parser.add_argument("--evaluated", default=DEFAULT_EVALUATED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    paths = write_replay_diagnostics_outputs(args.evaluated, args.output_dir)
    print(NOTICE)
    print(f"evaluated path: {Path(args.evaluated)}")
    print(f"report path: {paths['report']}")
    print(f"overall path: {paths['overall']}")
    print(f"by tier path: {paths['by_tier']}")
    print(f"by score bucket path: {paths['by_score_bucket']}")
    print(f"by signal label path: {paths['by_signal_label']}")
    print(f"by replay date path: {paths['by_replay_date']}")


if __name__ == "__main__":
    main()
