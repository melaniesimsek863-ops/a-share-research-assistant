from __future__ import annotations

from datetime import date

import pandas as pd

from a_share_ai.data.cache import CsvMarketCache
from a_share_ai.data.quality import CACHE_REFRESH_FAILED_COLUMN
from a_share_ai.data.schema import REQUIRED_BAR_COLUMNS
from a_share_ai.models import DATA_MODE_REAL_PUBLIC


class CachedMarketDataProvider:
    data_mode = DATA_MODE_REAL_PUBLIC
    provider_name = "akshare-cached"

    def __init__(
        self,
        inner: object,
        cache: CsvMarketCache,
        report_date: date,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.report_date = report_date

    def get_stock_info(self) -> pd.DataFrame:
        return self.inner.get_stock_info()

    def get_price_history(self, codes: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for code in codes:
            cached = self.cache.read_daily(code)
            if _covers_report_date(cached, self.report_date):
                frames.append(cached)
                continue
            try:
                remote = self.inner.get_price_history([code])
            except Exception as exc:
                self.cache.update_manifest_failure(code, str(exc))
                frames.append(_failed_refresh_fallback(cached))
                continue
            if remote.empty:
                self.cache.update_manifest_failure(code, "empty remote price history")
                frames.append(_failed_refresh_fallback(cached))
                continue
            merged = _merge_bars(cached, remote)
            self.cache.write_daily(code, merged)
            refreshed = self.cache.read_daily(code)
            self.cache.update_manifest_success(code, _latest_date(refreshed))
            frames.append(refreshed)
        if not frames:
            return _empty_bars()
        return pd.concat(frames, ignore_index=True)

    def get_index_history(self) -> pd.DataFrame:
        return self.inner.get_index_history()


def _covers_report_date(frame: pd.DataFrame, report_date: date) -> bool:
    latest = _latest_date(frame)
    return latest is not None and latest >= report_date


def _latest_date(frame: pd.DataFrame) -> date | None:
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date.dropna()
    return None if dates.empty else max(dates)


def _merge_bars(cached: pd.DataFrame, remote: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([cached, remote], ignore_index=True)
    if combined.empty:
        return _empty_bars()
    return (
        combined.drop_duplicates(["date", "code"], keep="last")
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )


def _failed_refresh_fallback(cached: pd.DataFrame) -> pd.DataFrame:
    if cached.empty:
        return _empty_bars()
    fallback = cached.copy()
    fallback[CACHE_REFRESH_FAILED_COLUMN] = True
    return fallback



def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS)
