from __future__ import annotations

import argparse
from datetime import date

from a_share_ai.data.providers import DataProviderError
from a_share_ai.pipeline import run_daily_pipeline
from a_share_ai.models import RiskState


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share AI trading assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("daily", help="Generate daily research report")
    daily.add_argument("--config", default="configs/default.yaml")
    daily.add_argument("--date", required=True)
    daily.add_argument("--provider", choices=["demo", "akshare"])

    daily.add_argument("--current-drawdown", type=float)
    daily.add_argument("--current-holdings", type=int)
    daily.add_argument("--daily-new-buys", type=int)
    args = parser.parse_args()
    if args.command == "daily":
        supplied_risk_values = (
            args.current_drawdown,
            args.current_holdings,
            args.daily_new_buys,
        )
        risk_state = None
        if any(value is not None for value in supplied_risk_values):
            risk_state = RiskState(
                current_drawdown=args.current_drawdown or 0.0,
                current_holdings=args.current_holdings or 0,
                daily_new_buys=args.daily_new_buys or 0,
            )
        try:
            result = run_daily_pipeline(
                args.config,
                date.fromisoformat(args.date),
                provider_name=args.provider,
                risk_state=risk_state,
            )
        except DataProviderError as exc:
            parser.exit(status=2, message=f"Data provider failed: {exc}\n")
        print(f"Markdown report: {result['markdown_path']}")
        print(f"Excel signals: {result['excel_path']}")
