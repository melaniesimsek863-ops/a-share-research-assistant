from __future__ import annotations

from datetime import date

import pandas as pd

from a_share_ai.models import BOARD_STAR, UniverseConfig


def _exclusion_reason(
    row: pd.Series,
    as_of: date,
    config: UniverseConfig,
    check_liquidity: bool,
) -> str | None:
    if config.exclude_star_market and row["board"] == BOARD_STAR:
        return "star_market_excluded"
    if row["board"] not in config.include_boards:
        return "board_not_included"
    if config.exclude_st and bool(row["is_st"]):
        return "st_or_delisting_risk"
    if config.exclude_delisting_risk and bool(row["is_delisting_risk"]):
        return "st_or_delisting_risk"
    listing_days = (as_of - row["listing_date"]).days
    if listing_days < config.min_listing_days:
        return "listed_less_than_120_days"
    if config.exclude_suspended and bool(row["is_suspended"]):
        return "suspended"
    if check_liquidity and pd.isna(row["avg_turnover_20d"]):
        return "missing_liquidity_data"
    if check_liquidity and (
        float(row["avg_turnover_20d"]) < config.min_avg_turnover_20d
    ):
        return "low_liquidity"
    return None


def filter_universe(
    stocks: pd.DataFrame,
    as_of: date,
    config: UniverseConfig,
    check_liquidity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "code",
        "name",
        "board",
        "listing_date",
        "is_st",
        "is_delisting_risk",
        "is_suspended",
        "avg_turnover_20d",
    }
    missing = required - set(stocks.columns)
    if missing:
        raise ValueError(f"stocks missing required columns: {sorted(missing)}")

    evaluated = stocks.copy()
    evaluated["exclude_reason"] = [
        _exclusion_reason(row, as_of, config, check_liquidity)
        for _, row in evaluated.iterrows()
    ]
    eligible = evaluated[evaluated["exclude_reason"].isna()].drop(columns=["exclude_reason"])
    excluded = evaluated[evaluated["exclude_reason"].notna()]
    return eligible.reset_index(drop=True), excluded.reset_index(drop=True)
