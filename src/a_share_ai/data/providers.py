from __future__ import annotations

from datetime import date
from typing import Protocol


import pandas as pd

from a_share_ai.data.demo import demo_price_history, demo_stock_info


class DataProviderError(RuntimeError):
    pass


class MarketDataProvider(Protocol):
    data_mode: str

    def get_stock_info(self) -> pd.DataFrame:
        ...

    def get_price_history(self, codes: list[str]) -> pd.DataFrame:
        ...

    def get_index_history(self) -> pd.DataFrame:
        ...


class LocalSampleProvider:
    data_mode = "synthetic_demo"

    def get_stock_info(self) -> pd.DataFrame:
        return demo_stock_info()

    def get_price_history(self, codes: list[str]) -> pd.DataFrame:
        if not codes:
            return demo_price_history(days=0)
        frames = [demo_price_history(code=code, days=140, trend=0.002) for code in codes]
        return pd.concat(frames, ignore_index=True)

    def get_index_history(self) -> pd.DataFrame:
        return demo_price_history(code="000300", days=140, trend=0.001)


def create_provider(
    name: str,
    adjustment: str = "qfq",
    *,
    cache_enabled: bool = False,
    cache_base_dir: str = "data/raw/akshare",
    report_date: date | None = None,
) -> MarketDataProvider:
    if adjustment != "qfq":
        raise ValueError(
            "Provider adjustment must be 'qfq' for v0.3 real-data ingestion"
        )
    normalized = name.strip().lower()
    if normalized == "demo":
        return LocalSampleProvider()
    if normalized == "akshare":
        from a_share_ai.data.akshare_provider import AkShareProvider

        provider = AkShareProvider(adjustment=adjustment)
        if cache_enabled:
            if report_date is None:
                raise ValueError("report_date is required when akshare cache is enabled")
            from a_share_ai.data.cache import CsvMarketCache
            from a_share_ai.data.cached_provider import CachedMarketDataProvider

            return CachedMarketDataProvider(
                provider,
                CsvMarketCache(cache_base_dir),
                report_date=report_date,
            )
        return provider
    raise ValueError(f"Unsupported provider: {name}")
