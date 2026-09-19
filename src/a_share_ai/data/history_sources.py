from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import pandas as pd

from a_share_ai.data.providers import DataProviderError
from a_share_ai.data.schema import REQUIRED_BAR_COLUMNS, normalize_price_history


def fetch_stock_daily_with_fallbacks(
    akshare_module: object,
    code: str,
    adjustment: str,
    http_get: Callable[..., object] | None = None,
) -> pd.DataFrame:
    if adjustment != "qfq":
        raise DataProviderError("history source fallbacks support only qfq adjustment")

    failures: list[str] = []
    for name, fetch in _history_sources(akshare_module, code, adjustment, http_get):
        try:
            raw_frame = fetch()
            frame = normalize_price_history(raw_frame, code=code)
            _validate_normalized_history(name, frame)
            return frame
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    raise DataProviderError("All history sources failed: " + " | ".join(failures))


def _history_sources(
    akshare_module: object,
    code: str,
    adjustment: str,
    http_get: Callable[..., object] | None,
) -> list[tuple[str, Callable[[], pd.DataFrame]]]:
    sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        (
            "stock_zh_a_hist",
            lambda: akshare_module.stock_zh_a_hist(
                symbol=code,
                period="daily",
                adjust=adjustment,
            ),
        )
    ]
    if http_get is not None:
        sources.append(
            ("eastmoney_raw", lambda: eastmoney_hist_raw(code, adjustment, http_get))
        )
    optional_tx = getattr(akshare_module, "stock_zh_a_hist_tx", None)
    if callable(optional_tx):
        sources.append(
            (
                "stock_zh_a_hist_tx",
                lambda: optional_tx(symbol=code, adjust=adjustment),
            )
        )
    return sources


def eastmoney_hist_raw(
    code: str,
    adjustment: str,
    http_get: Callable[..., object],
) -> pd.DataFrame:
    secid = _eastmoney_market_code(code)
    fqt = _eastmoney_fqt(adjustment)
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        "?fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt={fqt}&secid={secid}&beg=19900101&end=20500101"
    )
    response = http_get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return normalize_eastmoney_hist_json(payload)


def normalize_eastmoney_hist_json(payload: dict[str, object]) -> pd.DataFrame:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise DataProviderError("Eastmoney history fallback returned no data object")
    klines = data.get("klines")
    if not isinstance(klines, list):
        raise DataProviderError("Eastmoney history fallback returned no kline rows")

    rows: list[dict[str, object]] = []
    for raw_line in klines:
        parts = str(raw_line).split(",")
        if len(parts) < 7:
            continue
        rows.append(
            {
                "\u65e5\u671f": parts[0],
                "\u5f00\u76d8": float(parts[1]),
                "\u6536\u76d8": float(parts[2]),
                "\u6700\u9ad8": float(parts[3]),
                "\u6700\u4f4e": float(parts[4]),
                "\u6210\u4ea4\u91cf": float(parts[5]),
                "\u6210\u4ea4\u989d": float(parts[6]),
            }
        )
    if not rows:
        raise DataProviderError("Eastmoney history fallback returned no parseable kline rows")
    return pd.DataFrame(rows)


def _validate_normalized_history(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise DataProviderError(f"{name} returned empty normalized history")

    missing_columns = [column for column in REQUIRED_BAR_COLUMNS if column not in frame]
    if missing_columns:
        raise DataProviderError(
            f"{name} returned history without required fields: {','.join(missing_columns)}"
        )

    unusable_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "amount")
        if not frame[column].map(_is_finite_number).all()
    ]
    if unusable_columns:
        raise DataProviderError(
            f"{name} returned history with unusable fields: {','.join(unusable_columns)}"
        )

    if frame["date"].isna().any() or frame["code"].isna().any():
        raise DataProviderError(f"{name} returned history with unusable date or code")
    if frame["code"].astype(str).str.strip().eq("").any():
        raise DataProviderError(f"{name} returned history with unusable date or code")


def _is_finite_number(value: object) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _eastmoney_market_code(code: str) -> str:
    normalized = _normalize_code(code)
    if normalized.startswith(("600", "601", "603", "605", "688")):
        return "1." + normalized
    if normalized.startswith(("000", "001", "002", "003", "300", "301")):
        return "0." + normalized
    raise DataProviderError(f"Unsupported A-share code for Eastmoney history: {code}")


def _eastmoney_fqt(adjustment: str) -> str:
    if adjustment == "qfq":
        return "1"
    raise DataProviderError("Eastmoney history fallback supports only qfq adjustment")


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) < 6 else text
