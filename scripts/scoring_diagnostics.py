from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_ai.config import load_config
from a_share_ai.diagnostics.scoring import write_scoring_diagnostics_outputs

DEFAULT_EVALUATED = "outputs/replay-cached-history-resume-60d-20260803/2026-07-31_replay_evaluated.csv"
DEFAULT_OUTPUT_DIR = "outputs/scoring-diagnostics"
NOTICE = "SCORING DIAGNOSTICS / NOT A RECOMMENDATION"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scoring and tier diagnostics from an evaluated replay CSV")
    parser.add_argument("--evaluated", default=DEFAULT_EVALUATED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    strategy_config = load_config(args.config).strategy if args.experimental else None
    paths = write_scoring_diagnostics_outputs(
        args.evaluated,
        args.output_dir,
        include_experimental=args.experimental,
        strategy_config=strategy_config,
    )
    print(NOTICE)
    print(f"evaluated path: {Path(args.evaluated)}")
    print(f"report path: {paths['report']}")
    print(f"tier delta path: {paths['tier_delta']}")
    print(f"score factor buckets path: {paths['score_factor_buckets']}")
    print(f"signal label diagnostics path: {paths['signal_label_diagnostics']}")
    print(f"signal combo diagnostics path: {paths['signal_combo_diagnostics']}")
    print(f"bad replay dates path: {paths['bad_replay_dates']}")
    if args.experimental:
        print("experimental scoring comparison: HISTORICAL RESEARCH ONLY / NOT A RECOMMENDATION")
        print(f"experimental report path: {paths['experimental_report']}")
        print(f"experimental tier delta path: {paths['experimental_tier_delta']}")
        print(f"experimental factor buckets path: {paths['experimental_factor_buckets']}")
        print(f"experimental overlap path: {paths['experimental_overlap']}")


if __name__ == "__main__":
    main()