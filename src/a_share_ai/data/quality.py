from __future__ import annotations

from datetime import date

import pandas as pd

from a_share_ai.data.schema import REQUIRED_BAR_COLUMNS
from a_share_ai.models import (
    DataQualityStatus,
    QUALITY_BLOCKED,
    QUALITY_OK,
    QUALITY_WARNING,
)

CACHE_REFRESH_FAILED_COLUMN = "cache_refresh_failed"


def latest_data_date(price_history: pd.DataFrame) -> date | None:
    if price_history.empty or "date" not in price_history.columns:
        return None
    dates = pd.to_datetime(price_history["date"], errors="coerce").dropna()
    return None if dates.empty else dates.max().date()


def remove_future_bars(
    price_history: pd.DataFrame, report_date: date
) -> tuple[pd.DataFrame, bool]:
    if price_history.empty or "date" not in price_history.columns:
        return price_history.copy(), False

    parsed_dates = pd.to_datetime(price_history["date"], errors="coerce")
    future_mask = parsed_dates.notna() & (parsed_dates.dt.date > report_date)
    return price_history.loc[~future_mask].copy(), bool(future_mask.any())


_NUMERIC_BAR_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def _empty_standard_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=list(REQUIRED_BAR_COLUMNS))


def validate_bar_data(
    price_history: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate and clean standard daily bars before feature computation."""
    missing = sorted(set(REQUIRED_BAR_COLUMNS) - set(price_history.columns))
    if missing:
        return (
            _empty_standard_bars(),
            [],
            [f"missing_bar_columns:{','.join(missing)}"],
        )
    if price_history.empty:
        return _empty_standard_bars(), [], []

    data = price_history.copy()
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    codes = data["code"].fillna("").astype(str).str.strip()
    numeric: dict[str, pd.Series] = {}
    unusable: list[str] = []
    for column in _NUMERIC_BAR_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce").replace(
            [float("inf"), float("-inf")], float("nan")
        )
        numeric[column] = values
        if not values.notna().any():
            unusable.append(column)

    if unusable:
        return (
            _empty_standard_bars(),
            [],
            [f"unusable_bar_columns:{','.join(sorted(unusable))}"],
        )

    valid_mask = parsed_dates.notna() & codes.ne("")
    for column in _NUMERIC_BAR_COLUMNS:
        valid_mask &= numeric[column].notna()

    warnings: list[str] = []
    removed_count = int((~valid_mask).sum())
    if removed_count:
        warnings.append(f"invalid_bar_rows_removed:{removed_count}")

    cleaned = data.loc[valid_mask, list(REQUIRED_BAR_COLUMNS)].copy()
    if cleaned.empty:
        return _empty_standard_bars(), warnings, ["no_usable_bar_rows"]

    cleaned["date"] = parsed_dates.loc[valid_mask].dt.date
    cleaned["code"] = codes.loc[valid_mask]
    for column in _NUMERIC_BAR_COLUMNS:
        cleaned[column] = numeric[column].loc[valid_mask]
    return cleaned.reset_index(drop=True), warnings, []


def assess_data_quality(
    provider: str,
    data_mode: str,
    price_history: pd.DataFrame,
    report_date: date,
    stale_days: int,
    adjustment: str,
    warnings: list[str] | None = None,
    blocking_reasons: list[str] | None = None,
    source_had_future: bool = False,
) -> DataQualityStatus:
    filtered_history, validated_had_future = remove_future_bars(
        price_history, report_date
    )
    had_future = source_had_future or validated_had_future
    latest_date = latest_data_date(filtered_history)
    quality_warnings = list(warnings or [])
    quality_blocking_reasons = list(blocking_reasons or [])

    if had_future and "future_data_removed" not in quality_warnings:
        quality_warnings.append("future_data_removed")
    if had_future and "future_data_detected" not in quality_blocking_reasons:
        quality_blocking_reasons.append("future_data_detected")

    if latest_date is None:
        if "no_price_data" not in quality_blocking_reasons:
            quality_blocking_reasons.append("no_price_data")
        is_stale = False
    else:
        is_stale = (report_date - latest_date).days > stale_days
        if is_stale and "stale_data" not in quality_blocking_reasons:
            quality_blocking_reasons.append("stale_data")

    if quality_blocking_reasons:
        quality_status = QUALITY_BLOCKED
    elif quality_warnings:
        quality_status = QUALITY_WARNING
    else:
        quality_status = QUALITY_OK

    return DataQualityStatus(
        provider=provider,
        data_mode=data_mode,
        latest_data_date=latest_date,
        report_date=report_date,
        is_stale=is_stale,
        is_future_dated=had_future,
        quality_status=quality_status,
        warnings=quality_warnings,
        blocking_reasons=quality_blocking_reasons,
        adjustment=adjustment,
    )


def apply_quality_block(
    candidates: pd.DataFrame, status: DataQualityStatus
) -> pd.DataFrame:
    result = candidates.copy()
    if status.quality_status != QUALITY_BLOCKED:
        return result

    result["can_buy"] = pd.Series([False] * len(result), index=result.index, dtype=object)
    result["risk_action"] = "data_quality_blocked"
    reasons = ",".join(status.blocking_reasons)
    result["risk_notes"] = f"数据质量阻塞：{reasons}"
    result["suggested_position"] = 0.0
    return result
