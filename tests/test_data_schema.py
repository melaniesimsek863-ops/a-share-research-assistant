from datetime import date

import pandas as pd

from a_share_ai.data.schema import (
    infer_board,
    normalize_index_history,
    normalize_price_history,
    normalize_stock_info,
)
from a_share_ai.models import BOARD_CHINEXT, BOARD_MAIN, BOARD_STAR


def test_infer_board_from_code_prefix() -> None:
    assert infer_board("600000") == BOARD_MAIN
    assert infer_board("000001") == BOARD_MAIN
    assert infer_board("300750") == BOARD_CHINEXT
    assert infer_board("688001") == BOARD_STAR
    assert infer_board("900001") == "unknown"


def test_normalize_stock_info_maps_required_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "代码": "300750",
                "名称": "宁德时代",
                "上市时间": "2018-06-11",
                "是否停牌": False,
                "20日平均成交额": 2500000000,
            }
        ]
    )

    result = normalize_stock_info(raw)

    assert result.loc[0, "code"] == "300750"
    assert result.loc[0, "name"] == "宁德时代"
    assert result.loc[0, "board"] == BOARD_CHINEXT
    assert result.loc[0, "listing_date"] == date(2018, 6, 11)
    assert bool(result.loc[0, "is_st"]) is False
    assert bool(result.loc[0, "is_delisting_risk"]) is False
    assert bool(result.loc[0, "is_suspended"]) is False
    assert result.loc[0, "avg_turnover_20d"] == 2500000000


def test_normalize_price_history_maps_ohlcv_amount() -> None:
    raw = pd.DataFrame(
        [
            {
                "日期": "2026-07-28",
                "开盘": "10.1",
                "最高": "10.8",
                "最低": "9.9",
                "收盘": "10.5",
                "成交量": "1000000",
                "成交额": "10500000",
            }
        ]
    )

    result = normalize_price_history(raw, code="000001")

    assert result.loc[0, "date"] == date(2026, 7, 28)
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "close"] == 10.5
    assert result.loc[0, "amount"] == 10500000


def test_normalize_index_history_supports_public_bar_contract() -> None:
    raw = pd.DataFrame(
        [{"date": "2026-07-28", "close": "3400.5", "volume": "10", "amount": "20"}],
    )

    result = normalize_index_history(raw, code="000001")

    assert list(result.columns) == [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "close"] == 3400.5


def test_normalize_standard_english_price_columns() -> None:
    raw = pd.DataFrame(
        [{
            "date": "2026-07-28",
            "code": "000001",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
            "amount": 1050,
        }]
    )

    result = normalize_price_history(raw)

    assert result.loc[0, "date"] == date(2026, 7, 28)
    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "close"] == 10.5


def test_normalize_stock_info_zero_pads_numeric_code() -> None:
    raw = pd.DataFrame([{"code": 1, "name": "name"}])
    result = normalize_stock_info(raw)


    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "board"] == BOARD_MAIN


def test_normalize_index_history_marks_missing_amount_as_unavailable_zero() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-07-28",
                "open": 3400,
                "high": 3410,
                "low": 3390,
                "close": 3400.5,
                "volume": 10,
            }
        ]
    )

    result = normalize_index_history(raw, code="000001")

    assert result.loc[0, "amount"] == 0.0
