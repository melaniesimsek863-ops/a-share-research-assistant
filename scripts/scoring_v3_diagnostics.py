from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.diagnostics.scoring_v3 import write_scoring_v3_diagnostics_outputs
from a_share_ai.models import StrategyConfig

DEFAULT_EVALUATED = "outputs/replay-cached-history-resume-60d-20260803/2026-07-31_replay_evaluated.csv"
DEFAULT_OUTPUT_DIR = "outputs/scoring-v3-diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build experimental v2 vs v3 scoring diagnostics from an evaluated replay CSV")
    parser.add_argument("--evaluated", default=DEFAULT_EVALUATED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--focused-count", type=int, default=StrategyConfig.focused_count)
    parser.add_argument("--candidate-count", type=int, default=StrategyConfig.candidate_count)
    args = parser.parse_args()
    strategy_config = StrategyConfig(focused_count=args.focused_count, candidate_count=args.candidate_count)
    paths = write_scoring_v3_diagnostics_outputs(args.evaluated, args.output_dir, strategy_config)
    print("v3 diagnostics: HISTORICAL RESEARCH ONLY / NOT A RECOMMENDATION")
    print(f"evaluated path: {Path(args.evaluated)}")
    print(f"v3 report path: {paths['v3_report']}")
    print(f"factor report path: {paths['factor_report']}")
    print(f"v3 robustness summary path: {paths['v3_summary']}")


if __name__ == "__main__":
    main()
