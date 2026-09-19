from datetime import date

import pandas as pd
import pytest

from a_share_ai.data.akshare_provider import (
    AkShareProvider,
    _DefaultAkShareClient,
    _build_base_stock_info,
)
from a_share_ai.data.providers import DataProviderError, LocalSampleProvider, create_provider


class FakeAkShareClient:
    def stock_info(self):
        return pd.DataFrame([{"代码": "000001", "名称": "平安银行", "上市时间": "1991-04-03"}])

    def stock_daily(self, code: str, adjustment: str):
        assert code == "000001"
        assert adjustment == "qfq"
        return pd.DataFrame(
            [{"日期": "2026-07-28", "开盘": 10, "最高": 11, "最低": 9, "收盘": 10.5, "成交量": 1, "成交额": 10.5}]
        )

    def index_daily(self, code: str):
        return pd.DataFrame(
            [{"日期": "2026-07-28", "开盘": 100, "最高": 110, "最低": 90, "收盘": 105, "成交量": 1, "成交额": 105}]
        )


def test_akshare_provider_normalizes_fake_client_frames() -> None:
    provider = AkShareProvider(akshare_client=FakeAkShareClient(), adjustment="qfq")

    stocks = provider.get_stock_info()
    prices = provider.get_price_history(["000001"])
    indexes = provider.get_index_history()

    assert stocks.loc[0, "code"] == "000001"
    assert prices.loc[0, "code"] == "000001"
    assert set(indexes["code"]) == {"000300", "000001.SH", "399006.SZ"}


def test_akshare_provider_wraps_client_errors() -> None:
    class BrokenClient:
        def stock_info(self):
            raise ValueError("remote changed")

    provider = AkShareProvider(akshare_client=BrokenClient())

    try:
        provider.get_stock_info()
    except DataProviderError as exc:
        assert "stock_info" in str(exc)
    else:
        raise AssertionError("expected DataProviderError")


def test_default_akshare_client_uses_csi_symbol_for_csi_300() -> None:
    class FakeAkShareModule:
        def __init__(self) -> None:
            self.index_symbols: list[str] = []

        def stock_zh_index_daily(self, symbol: str) -> pd.DataFrame:
            self.index_symbols.append(symbol)
            return pd.DataFrame()

    fake_module = FakeAkShareModule()

    _DefaultAkShareClient(fake_module).index_daily("000300")

    assert fake_module.index_symbols == ["csi000300"]


def test_default_akshare_client_continues_when_spot_source_fails() -> None:
    class FakeAkShareModule:
        def stock_info_a_code_name(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"code": "000001", "name": "平安银行"},
                    {"code": "600000", "name": "浦发银行"},
                ]
            )

        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            raise ConnectionError("82.push2.eastmoney.com unavailable")

        def stock_info_sh_name_code(self, symbol: str) -> pd.DataFrame:
            assert symbol in {"主板A股", "科创板"}
            if symbol == "科创板":
                return pd.DataFrame(columns=["证券代码", "证券简称", "上市日期"])
            return pd.DataFrame(
                [{"证券代码": "600000", "证券简称": "浦发银行", "上市日期": "1999-11-10"}]
            )

        def stock_info_sz_name_code(self, symbol: str) -> pd.DataFrame:
            assert symbol == "A股列表"
            return pd.DataFrame(
                [{"A股代码": "000001", "A股简称": "平安银行", "A股上市日期": "1991-04-03"}]
            )

    result = _DefaultAkShareClient(FakeAkShareModule()).stock_info()

    assert result["code"].tolist() == ["000001", "600000"]
    assert result["listing_date"].tolist() == [date(1991, 4, 3), date(1999, 11, 10)]
    assert result["is_suspended"].tolist() == [False, False]
    assert result["avg_turnover_20d"].isna().all()


@pytest.mark.parametrize(
    ("stock_rows", "listing_rows", "message"),
    [
        ([], [], "code/name"),
        (
            [{"code": "000001", "name": ""}],
            [{"code": "000001", "listing_date": "1991-04-03"}],
            "code/name",
        ),
        (
            [{"code": "000001", "name": "bank"}],
            [{"code": "000001", "listing_date": "not-a-date"}],
            "listing_date",
        ),
        (
            [{"code": "", "name": "bank"}],
            [],
            "code/name",
        ),
    ],
)
def test_build_base_stock_info_rejects_invalid_metadata_values(
    stock_rows, listing_rows, message
) -> None:
    """Removing value validation must reject empty or unparsable A-share metadata."""

    class FakeAkShareModule:
        def stock_info_a_code_name(self) -> pd.DataFrame:
            return pd.DataFrame(stock_rows, columns=["code", "name"])

        def stock_info_sz_name_code(self, symbol: str) -> pd.DataFrame:
            return pd.DataFrame(listing_rows, columns=["code", "listing_date"])

    with pytest.raises(DataProviderError, match=message):
        _build_base_stock_info(FakeAkShareModule())
def test_create_provider_demo() -> None:
    """Changing the demo branch must not replace the deterministic offline provider."""
    assert isinstance(create_provider("demo"), LocalSampleProvider)


def test_create_provider_wraps_akshare_when_cache_enabled(monkeypatch, tmp_path) -> None:
    class FakeAkShareProvider:
        data_mode = "real_public_market_data"

        def __init__(self, adjustment: str = "qfq") -> None:
            self.adjustment = adjustment

    import a_share_ai.data.akshare_provider as akshare_provider

    monkeypatch.setattr(akshare_provider, "AkShareProvider", FakeAkShareProvider)

    provider = create_provider(
        "akshare",
        adjustment="qfq",
        cache_enabled=True,
        cache_base_dir=str(tmp_path / "akshare"),
        report_date=date(2026, 7, 30),
    )

    assert provider.provider_name == "akshare-cached"


def test_create_provider_rejects_unknown() -> None:
    """Unknown provider names must be rejected instead of falling back to demo data."""
    try:
        create_provider("unknown")
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_default_akshare_client_combines_documented_spot_and_listing_sources() -> None:
    class FakeAkShareModule:
        def stock_info_a_code_name(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"code": "000001", "name": "\u5e73\u5b89\u94f6\u884c"},
                    {"code": "600000", "name": "\u6d66\u53d1\u94f6\u884c"},
                ]
            )

        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "\u4ee3\u7801": "000001",
                        "\u540d\u79f0": "\u5e73\u5b89\u94f6\u884c",
                        "\u6700\u65b0\u4ef7": 10.5,
                        "\u6210\u4ea4\u91cf": 1_000_000,
                        "\u6210\u4ea4\u989d": 10_500_000,
                    },
                    {
                        "\u4ee3\u7801": "600000",
                        "\u540d\u79f0": "\u6d66\u53d1\u94f6\u884c",
                        "\u6700\u65b0\u4ef7": None,
                        "\u6210\u4ea4\u91cf": 0,
                        "\u6210\u4ea4\u989d": 0,
                    },
                ]
            )

        def stock_info_sh_name_code(self, symbol: str) -> pd.DataFrame:
            assert symbol in {"\u4e3b\u677fA\u80a1", "\u79d1\u521b\u677f"}
            if symbol == "\u79d1\u521b\u677f":
                return pd.DataFrame(
                    columns=["\u8bc1\u5238\u4ee3\u7801", "\u8bc1\u5238\u7b80\u79f0", "\u4e0a\u5e02\u65e5\u671f"]
                )
            return pd.DataFrame(
                [{
                    "\u8bc1\u5238\u4ee3\u7801": "600000",
                    "\u8bc1\u5238\u7b80\u79f0": "\u6d66\u53d1\u94f6\u884c",
                    "\u4e0a\u5e02\u65e5\u671f": "1999-11-10",
                }]
            )

        def stock_info_sz_name_code(self, symbol: str) -> pd.DataFrame:
            assert symbol == "A\u80a1\u5217\u8868"
            return pd.DataFrame(
                [{
                    "A\u80a1\u4ee3\u7801": "000001",
                    "A\u80a1\u7b80\u79f0": "\u5e73\u5b89\u94f6\u884c",
                    "A\u80a1\u4e0a\u5e02\u65e5\u671f": "1991-04-03",
                }]
            )

    result = _DefaultAkShareClient(FakeAkShareModule()).stock_info()

    assert result["code"].tolist() == ["000001", "600000"]
    assert result["listing_date"].tolist() == [date(1991, 4, 3), date(1999, 11, 10)]
    assert result["is_suspended"].tolist() == [False, True]
    assert result["avg_turnover_20d"].isna().all()


def test_default_akshare_client_raises_only_after_metadata_fallbacks_fail() -> None:
    class FakeAkShareModule:
        def stock_info_a_code_name(self) -> pd.DataFrame:
            return pd.DataFrame([{"code": "000001", "name": "bank"}])

        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            return pd.DataFrame(
                [{
                    "\u4ee3\u7801": "000001",
                    "\u540d\u79f0": "bank",
                    "\u6700\u65b0\u4ef7": 10,
                    "\u6210\u4ea4\u91cf": 1,
                    "\u6210\u4ea4\u989d": 10,
                }]
            )

        def stock_individual_info_em(self, symbol: str) -> pd.DataFrame:
            assert symbol == "000001"
            return pd.DataFrame([{"item": "industry", "value": "bank"}])

    with pytest.raises(DataProviderError, match="listing_date"):
        _DefaultAkShareClient(FakeAkShareModule()).stock_info()


def test_create_provider_rejects_non_qfq_adjustment() -> None:
    with pytest.raises(ValueError, match="qfq"):
        create_provider("akshare", adjustment="hfq")


def test_normalized_index_amount_is_explicitly_unavailable() -> None:
    class IndexWithoutAmountClient(FakeAkShareClient):
        def index_daily(self, code: str):
            return pd.DataFrame(
                [{
                    "\u65e5\u671f": "2026-07-28",
                    "\u5f00\u76d8": 100,
                    "\u6700\u9ad8": 110,
                    "\u6700\u4f4e": 90,
                    "\u6536\u76d8": 105,
                    "\u6210\u4ea4\u91cf": 1,
                }]
            )

    indexes = AkShareProvider(
        akshare_client=IndexWithoutAmountClient(), adjustment="qfq"
    ).get_index_history()

    assert indexes["amount"].eq(0.0).all()



def test_default_akshare_client_rejects_invalid_individual_listing_date_fallback() -> None:
    class FakeAkShareModule:
        def stock_info_a_code_name(self) -> pd.DataFrame:
            return pd.DataFrame([{"code": "000001", "name": "bank"}])

        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            return pd.DataFrame(
                [{
                    "\u4ee3\u7801": "000001",
                    "\u540d\u79f0": "bank",
                    "\u6700\u65b0\u4ef7": 10,
                    "\u6210\u4ea4\u91cf": 1,
                    "\u6210\u4ea4\u989d": 10,
                }]
            )

        def stock_individual_info_em(self, symbol: str) -> pd.DataFrame:
            assert symbol == "000001"
            return pd.DataFrame(
                [{"item": "\u4e0a\u5e02\u65f6\u95f4", "value": "not-a-date"}]
            )

    with pytest.raises(DataProviderError, match="listing_date"):
        _DefaultAkShareClient(FakeAkShareModule()).stock_info()

def test_default_akshare_client_normalizes_individual_listing_date_fallback() -> None:
    class FakeAkShareModule:
        def stock_info_a_code_name(self) -> pd.DataFrame:
            return pd.DataFrame([{"code": "000001", "name": "bank"}])

        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            return pd.DataFrame(
                [{
                    "\u4ee3\u7801": "000001",
                    "\u540d\u79f0": "bank",
                    "\u6700\u65b0\u4ef7": 10,
                    "\u6210\u4ea4\u91cf": 1,
                    "\u6210\u4ea4\u989d": 10,
                }]
            )

        def stock_individual_info_em(self, symbol: str) -> pd.DataFrame:
            assert symbol == "000001"
            return pd.DataFrame(
                [{"item": "\u4e0a\u5e02\u65f6\u95f4", "value": 19910403}]
            )

    provider = AkShareProvider(
        akshare_client=_DefaultAkShareClient(FakeAkShareModule())
    )

    result = provider.get_stock_info()

    assert result.loc[0, "listing_date"] == date(1991, 4, 3)


def test_default_akshare_client_uses_raw_history_fallback_when_hist_wrapper_fails() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": {
                    "klines": [
                        "2026-07-28,10.00,10.50,11.00,9.80,1000,1050000.00,0,0,0,0",
                    ]
                }
            }

    class FakeSession:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str, timeout: int) -> FakeResponse:
            assert timeout == 20
            self.urls.append(url)
            return FakeResponse()

    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            assert symbol == "000001"
            assert period == "daily"
            assert adjust == "qfq"
            raise ConnectionError("akshare wrapper failed")

    session = FakeSession()
    result = _DefaultAkShareClient(FakeAkShareModule(), http_get=session.get).stock_daily(
        "000001",
        "qfq",
    )

    assert session.urls
    assert result.loc[0, "date"] == date(2026, 7, 28)
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "open"] == 10.0
    assert result.loc[0, "close"] == 10.5
    assert result.loc[0, "amount"] == 1050000.0

def test_history_source_chain_uses_stock_zh_a_hist_first() -> None:
    calls: list[str] = []

    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            calls.append("hist")
            assert symbol == "000001"
            assert period == "daily"
            assert adjust == "qfq"
            return pd.DataFrame(
                [{
                    "\u65e5\u671f": "2026-07-30",
                    "\u5f00\u76d8": 10,
                    "\u6700\u9ad8": 11,
                    "\u6700\u4f4e": 9,
                    "\u6536\u76d8": 10.5,
                    "\u6210\u4ea4\u91cf": 100,
                    "\u6210\u4ea4\u989d": 1050,
                }]
            )

    result = _DefaultAkShareClient(FakeAkShareModule()).stock_daily("000001", "qfq")

    assert calls == ["hist"]
    assert list(result.columns) == [
        "date", "code", "open", "high", "low", "close", "volume", "amount"
    ]
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "amount"] == 1050


def test_history_source_chain_falls_back_to_raw_eastmoney_after_hist_failure() -> None:
    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("hist endpoint reset")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": {
                    "klines": [
                        "2026-07-30,10,10.5,11,9,100,1050,0,0,0,0",
                    ]
                }
            }

    urls: list[str] = []

    def fake_get(url: str, timeout: int) -> FakeResponse:
        urls.append(url)
        assert "push2his.eastmoney.com" in url
        assert timeout == 20
        return FakeResponse()

    result = _DefaultAkShareClient(FakeAkShareModule(), http_get=fake_get).stock_daily(
        "000001", "qfq"
    )

    assert len(urls) == 1
    assert result.loc[0, "amount"] == 1050


def test_history_source_chain_uses_optional_module_fallback_after_eastmoney_failure() -> None:
    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("hist endpoint reset")

        def stock_zh_a_hist_tx(self, symbol: str, adjust: str) -> pd.DataFrame:
            assert symbol == "000001"
            assert adjust == "qfq"
            return pd.DataFrame(
                [{
                    "date": "2026-07-30",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 100,
                    "amount": 1050,
                }]
            )

    def failing_get(url: str, timeout: int) -> object:
        raise ConnectionError("raw eastmoney reset")

    result = _DefaultAkShareClient(FakeAkShareModule(), http_get=failing_get).stock_daily(
        "000001", "qfq"
    )

    assert result.loc[0, "amount"] == 1050


def test_history_source_chain_reports_all_attempted_sources_when_all_fail() -> None:
    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("hist endpoint reset")


        def stock_zh_a_hist_tx(self, symbol: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("tencent endpoint reset")
    def failing_get(url: str, timeout: int) -> object:
        raise ConnectionError("raw eastmoney reset")

    with pytest.raises(DataProviderError) as exc_info:
        _DefaultAkShareClient(FakeAkShareModule(), http_get=failing_get).stock_daily(
            "000001", "qfq"
        )

    message = str(exc_info.value)
    assert "stock_zh_a_hist" in message
    assert "eastmoney_raw" in message

    assert "stock_zh_a_hist_tx" in message

def test_history_source_chain_rejects_optional_fallback_without_amount() -> None:
    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("hist endpoint reset")

        def stock_zh_a_hist_tx(self, symbol: str, adjust: str) -> pd.DataFrame:
            return pd.DataFrame(
                [{
                    "date": "2026-07-30",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 100,
                }]
            )

    def failing_get(url: str, timeout: int) -> object:
        raise ConnectionError("raw eastmoney reset")

    with pytest.raises(DataProviderError, match="amount"):
        _DefaultAkShareClient(FakeAkShareModule(), http_get=failing_get).stock_daily(
            "000001", "qfq"
        )


@pytest.mark.parametrize(
    "amount",
    [None, "", "bad", float("nan")],
    ids=["none", "blank", "malformed", "nan"],
)
def test_history_source_chain_rejects_optional_fallback_with_unusable_amount(
    amount: object,
) -> None:
    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("hist endpoint reset")

        def stock_zh_a_hist_tx(self, symbol: str, adjust: str) -> pd.DataFrame:
            return pd.DataFrame(
                [{
                    "date": "2026-07-30",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 100,
                    "amount": amount,
                }]
            )

    def failing_get(url: str, timeout: int) -> object:
        raise ConnectionError("raw eastmoney reset")

    with pytest.raises(DataProviderError, match="amount") as exc_info:
        _DefaultAkShareClient(FakeAkShareModule(), http_get=failing_get).stock_daily(
            "000001", "qfq"
        )

    assert "stock_zh_a_hist_tx" in str(exc_info.value)


@pytest.mark.parametrize("missing_column", ["date", "open", "high", "low", "close", "volume"])
def test_history_source_chain_rejects_optional_fallback_with_incomplete_bar(
    missing_column: str,
) -> None:
    class FakeAkShareModule:
        def stock_zh_a_hist(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise ConnectionError("hist endpoint reset")

        def stock_zh_a_hist_tx(self, symbol: str, adjust: str) -> pd.DataFrame:
            row: dict[str, object] = {
                "date": "2026-07-30",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
                "amount": 1050,
            }
            row.pop(missing_column)
            return pd.DataFrame([row])

    def failing_get(url: str, timeout: int) -> object:
        raise ConnectionError("raw eastmoney reset")

    with pytest.raises(DataProviderError) as exc_info:
        _DefaultAkShareClient(FakeAkShareModule(), http_get=failing_get).stock_daily(
            "000001", "qfq"
        )

    assert "stock_zh_a_hist_tx" in str(exc_info.value)
