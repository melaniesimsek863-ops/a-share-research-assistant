from datetime import date

import pandas as pd

from a_share_ai.data.cache import CsvMarketCache
from a_share_ai.data.cached_provider import CachedMarketDataProvider
from a_share_ai.data.providers import DataProviderError
from a_share_ai.models import DATA_MODE_REAL_PUBLIC


def bar(code: str, day: date, close: float = 10.0) -> dict[str, object]:
    return {
        "date": day,
        "code": code,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1,
        "amount": close,
    }


class FakeProvider:
    data_mode = DATA_MODE_REAL_PUBLIC
    provider_name = "fake-akshare"

    def __init__(self, price_history: pd.DataFrame | Exception) -> None:
        self.price_history = price_history
        self.price_calls: list[list[str]] = []

    def get_stock_info(self) -> pd.DataFrame:
        return pd.DataFrame([{"code": "000001", "name": "bank"}])

    def get_price_history(self, codes: list[str]) -> pd.DataFrame:
        self.price_calls.append(codes)
        if isinstance(self.price_history, Exception):
            raise self.price_history
        return self.price_history

    def get_index_history(self) -> pd.DataFrame:
        return pd.DataFrame()


def test_cached_provider_uses_cache_when_it_covers_report_date(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    cache.write_daily("000001", pd.DataFrame([bar("000001", date(2026, 7, 30))]))
    inner = FakeProvider(pd.DataFrame([bar("000001", date(2026, 7, 31))]))
    provider = CachedMarketDataProvider(inner, cache, report_date=date(2026, 7, 30))

    result = provider.get_price_history(["000001"])

    assert inner.price_calls == []
    assert result["date"].tolist() == [date(2026, 7, 30)]


def test_cached_provider_refreshes_stale_cache_and_records_success(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    cache.write_daily("000001", pd.DataFrame([bar("000001", date(2026, 7, 29), 9.0)]))
    inner = FakeProvider(pd.DataFrame([bar("000001", date(2026, 7, 30), 10.0)]))
    provider = CachedMarketDataProvider(inner, cache, report_date=date(2026, 7, 30))

    result = provider.get_price_history(["000001"])
    manifest = cache.read_manifest()

    assert inner.price_calls == [["000001"]]
    assert result["date"].tolist() == [date(2026, 7, 29), date(2026, 7, 30)]
    assert manifest.loc[0, "status"] == "ok"
    assert manifest.loc[0, "latest_cached_date"] == date(2026, 7, 30)


def test_cached_provider_returns_cache_and_records_failure_when_remote_fails(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    cache.write_daily("000001", pd.DataFrame([bar("000001", date(2026, 7, 29), 9.0)]))
    inner = FakeProvider(DataProviderError("remote timeout"))
    provider = CachedMarketDataProvider(inner, cache, report_date=date(2026, 7, 30))

    result = provider.get_price_history(["000001"])
    manifest = cache.read_manifest()

    assert result["date"].tolist() == [date(2026, 7, 29)]
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "failure_reason"] == "remote timeout"


def test_cached_provider_returns_empty_bars_when_remote_fails_without_cache(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    inner = FakeProvider(DataProviderError("remote timeout"))
    provider = CachedMarketDataProvider(inner, cache, report_date=date(2026, 7, 30))

    result = provider.get_price_history(["000001"])

    assert result.empty
    assert list(result.columns) == ["date", "code", "open", "high", "low", "close", "volume", "amount"]


def test_cached_provider_treats_empty_remote_as_failure_and_preserves_stale_cache(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    cache.write_daily("000001", pd.DataFrame([bar("000001", date(2026, 7, 29), 9.0)]))
    inner = FakeProvider(pd.DataFrame())
    provider = CachedMarketDataProvider(inner, cache, report_date=date(2026, 7, 30))

    result = provider.get_price_history(["000001"])
    manifest = cache.read_manifest()

    assert result["date"].tolist() == [date(2026, 7, 29)]
    assert result["cache_refresh_failed"].tolist() == [True]
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "failure_reason"] == "empty remote price history"
    assert pd.isna(manifest.loc[0, "latest_cached_date"])


def test_cached_provider_treats_empty_remote_as_failure_without_cache(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    inner = FakeProvider(pd.DataFrame())
    provider = CachedMarketDataProvider(inner, cache, report_date=date(2026, 7, 30))

    result = provider.get_price_history(["000001"])
    manifest = cache.read_manifest()

    assert result.empty
    assert list(result.columns) == ["date", "code", "open", "high", "low", "close", "volume", "amount"]
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "failure_reason"] == "empty remote price history"
    assert pd.isna(manifest.loc[0, "latest_cached_date"])
