from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.config import load_config
from a_share_ai.diagnostics.robustness import write_scoring_robustness_outputs

DEFAULT_EVALUATED = "outputs/replay-cached-history-resume-60d-20260803/2026-07-31_replay_evaluated.csv"
DEFAULT_OUTPUT_DIR = "outputs/scoring-robustness"
DEFAULT_CONFIG = "configs/default.yaml"


def _resolve_default_config(config_path: str) -> Path:
    path = Path(config_path)
    return ROOT / path if config_path == DEFAULT_CONFIG else path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-date scoring robustness diagnostics from an evaluated replay CSV"
    )
    parser.add_argument("--evaluated", default=DEFAULT_EVALUATED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    strategy_config = load_config(_resolve_default_config(args.config)).strategy
    paths = write_scoring_robustness_outputs(
        args.evaluated,
        args.output_dir,
        strategy_config,
    )

    print("robustness report: HISTORICAL RESEARCH ONLY / NOT A RECOMMENDATION")
    print(f"evaluated path: {Path(args.evaluated)}")
    print(f"summary path: {paths['summary']}")
    print(f"by-date path: {paths['by_date']}")
    print(f"by-horizon path: {paths['by_horizon']}")
    print(f"quantile-monotonicity path: {paths['quantile_monotonicity']}")
    print(f"focused-overlap path: {paths['focused_overlap']}")
    print(f"flags path: {paths['flags']}")


if __name__ == "__main__":
    main()