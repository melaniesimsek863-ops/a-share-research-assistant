from __future__ import annotations

from datetime import date

import pandas as pd

from a_share_ai.models import BOARD_CHINEXT, BOARD_MAIN, BOARD_STAR


REQUIRED_STOCK_INFO_COLUMNS = (
    "code",
    "name",
    "board",
    "listing_date",
    "is_st",
    "is_delisting_risk",
    "is_suspended",
    "avg_turnover_20d",
)
REQUIRED_BAR_COLUMNS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

STOCK_INFO_RENAME = {
    "代码": "code",
    "证券代码": "code",
    "名称": "name",
    "证券简称": "name",
    "上市时间": "listing_date",
    "上市日期": "listing_date",
    "是否停牌": "is_suspended",
    "停牌": "is_suspended",
    "20日平均成交额": "avg_turnover_20d",
}

PRICE_RENAME = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def _to_date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.date


def _to_bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    return values.fillna(default).map(
        lambda value: str(value).strip().lower() in {"true", "1", "是", "停牌"}
    )

def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) < 6 else text


def _normalize_code_series(values: pd.Series) -> pd.Series:
    return values.map(_normalize_code)



def infer_board(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return BOARD_MAIN
    if code.startswith(("300", "301")):
        return BOARD_CHINEXT
    if code.startswith("688"):
        return BOARD_STAR
    return "unknown"


def normalize_stock_info(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.rename(columns=STOCK_INFO_RENAME).copy()
    row_count = len(data)

    code = _normalize_code_series(data.get("code", pd.Series([None] * row_count, index=data.index)))
    name = data.get("name", pd.Series([""] * row_count, index=data.index)).fillna("").astype(str)
    board = data.get("board", pd.Series([None] * row_count, index=data.index))
    board = board.where(board.notna(), code.map(infer_board))

    is_st = _to_bool_series(
        data.get("is_st", pd.Series([False] * row_count, index=data.index))
    ) | name.str.contains("ST", case=False, na=False)
    is_delisting_risk = _to_bool_series(
        data.get("is_delisting_risk", pd.Series([False] * row_count, index=data.index))
    ) | name.str.contains("退", na=False)

    result = pd.DataFrame(index=data.index)
    result["code"] = code
    result["name"] = name
    result["board"] = board
    result["listing_date"] = _to_date_series(
        data.get("listing_date", pd.Series([None] * row_count, index=data.index))
    )
    result["is_st"] = is_st
    result["is_delisting_risk"] = is_delisting_risk
    result["is_suspended"] = _to_bool_series(
        data.get("is_suspended", pd.Series([False] * row_count, index=data.index))
    )
    result["avg_turnover_20d"] = pd.to_numeric(
        data.get("avg_turnover_20d", pd.Series([None] * row_count, index=data.index)),
        errors="coerce",
    )
    return result.loc[:, REQUIRED_STOCK_INFO_COLUMNS].reset_index(drop=True)


def normalize_price_history(raw: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
    data = raw.rename(columns=PRICE_RENAME).copy()
    row_count = len(data)
    result = pd.DataFrame(index=data.index)
    result["date"] = _to_date_series(
        data.get("date", pd.Series([None] * row_count, index=data.index))
    )
    if code is not None:
        result["code"] = _normalize_code(code)
    else:
        result["code"] = _normalize_code_series(
            data.get("code", pd.Series([None] * row_count, index=data.index))
        )
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(
            data.get(column, pd.Series([None] * row_count, index=data.index)),
            errors="coerce",
        )
    result = result.dropna(subset=["date", "close"])
    return result.loc[:, REQUIRED_BAR_COLUMNS].reset_index(drop=True)


def normalize_index_history(raw: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
    result = normalize_price_history(raw, code=code)
    # The selected AkShare index endpoint has no amount field; keep the standard
    # schema with an explicit unavailable sentinel because index bars are not traded.
    result["amount"] = result["amount"].fillna(0.0)
    return result
