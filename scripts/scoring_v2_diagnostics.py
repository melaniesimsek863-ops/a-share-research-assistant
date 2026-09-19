from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.config import load_config
from a_share_ai.diagnostics.scoring_v2 import write_scoring_v2_diagnostics_outputs

DEFAULT_EVALUATED = "outputs/replay-cached-history-resume-60d-20260803/2026-07-31_replay_evaluated.csv"
DEFAULT_OUTPUT_DIR = "outputs/scoring-v2-diagnostics"
DEFAULT_CONFIG = "configs/default.yaml"


def _resolve_default_config(config_path: str) -> Path:
    path = Path(config_path)
    return ROOT / path if config_path == DEFAULT_CONFIG else path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build factor and experimental v2 scoring diagnostics from an evaluated replay CSV"
    )
    parser.add_argument("--evaluated", default=DEFAULT_EVALUATED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    strategy_config = load_config(_resolve_default_config(args.config)).strategy
    paths = write_scoring_v2_diagnostics_outputs(args.evaluated, args.output_dir, strategy_config)

    print("v2 diagnostics: HISTORICAL RESEARCH ONLY / NOT A RECOMMENDATION")
    print(f"evaluated path: {Path(args.evaluated)}")
    print(f"v2 report path: {paths['v2_report']}")
    print(f"factor report path: {paths['factor_report']}")
    print(f"v2 robustness summary path: {paths['v2_summary']}")


if __name__ == "__main__":
    main()
