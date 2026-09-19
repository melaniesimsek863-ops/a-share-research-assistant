from datetime import date

import pandas as pd

from a_share_ai.data.cache import CsvMarketCache


def test_csv_market_cache_writes_sorted_deduplicated_daily_bars(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")
    bars = pd.DataFrame(
        [
            {"date": date(2026, 7, 30), "code": "000001", "open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 20, "amount": 200},
            {"date": date(2026, 7, 29), "code": "000001", "open": 1, "high": 2, "low": 0.8, "close": 1.5, "volume": 10, "amount": 100},
            {"date": date(2026, 7, 30), "code": "000001", "open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 20, "amount": 200},
        ]
    )

    cache.write_daily("000001", bars)
    result = cache.read_daily("000001")

    assert result["date"].tolist() == [date(2026, 7, 29), date(2026, 7, 30)]
    assert result["code"].tolist() == ["000001", "000001"]
    assert result["amount"].tolist() == [100, 200]


def test_csv_market_cache_manifest_tracks_success_and_failure(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")

    cache.update_manifest_failure("000001", "remote timeout")
    failed = cache.read_manifest()
    assert failed.loc[0, "code"] == "000001"
    assert failed.loc[0, "status"] == "failed"
    assert failed.loc[0, "failure_reason"] == "remote timeout"
    assert failed.loc[0, "consecutive_failures"] == 1

    cache.update_manifest_success("000001", date(2026, 7, 30))
    succeeded = cache.read_manifest()
    assert succeeded.loc[0, "status"] == "ok"
    assert succeeded.loc[0, "latest_cached_date"] == date(2026, 7, 30)
    assert succeeded.loc[0, "consecutive_failures"] == 0


def test_manifest_normalizes_unpadded_code_for_repeated_failures(tmp_path) -> None:
    cache = CsvMarketCache(tmp_path / "akshare")

    cache.update_manifest_failure("1", "remote timeout")
    cache.update_manifest_failure("1", "remote timeout")

    manifest = cache.read_manifest()
    assert len(manifest) == 1
    assert manifest.loc[0, "code"] == "000001"
    assert manifest.loc[0, "consecutive_failures"] == 2
