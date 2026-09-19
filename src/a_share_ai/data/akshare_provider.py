from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from a_share_ai.data.history_sources import fetch_stock_daily_with_fallbacks
from a_share_ai.data.providers import DataProviderError
from a_share_ai.data.schema import (
    normalize_index_history,
    normalize_price_history,
    normalize_stock_info,
)
from a_share_ai.models import DATA_MODE_REAL_PUBLIC




class _DefaultAkShareClient:
    def __init__(
        self,
        akshare_module: object,
        http_get: Callable[..., object] | None = None,
    ) -> None:
        self._akshare = akshare_module
        self._http_get = http_get

    def stock_info(self) -> pd.DataFrame:
        base = _build_base_stock_info(self._akshare)
        spot_raw = _optional_source(self._akshare, "stock_zh_a_spot_em")
        enriched = _enrich_with_optional_spot(base, spot_raw)
        return enriched.loc[
            :, ["code", "name", "listing_date", "is_suspended", "avg_turnover_20d"]
        ]
    def stock_daily(self, code: str, adjustment: str) -> pd.DataFrame:
        return fetch_stock_daily_with_fallbacks(
            self._akshare,
            code,
            adjustment,
            http_get=self._http_get,
        )

    def index_daily(self, code: str) -> pd.DataFrame:
        symbol = {
            "000300": "csi000300",
            "000001.SH": "sh000001",
            "399006.SZ": "sz399006",
        }[code]
        return self._akshare.stock_zh_index_daily(symbol=symbol)


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


def _eastmoney_hist_raw(
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
    return _normalize_eastmoney_hist_json(payload)


def _normalize_eastmoney_hist_json(payload: dict[str, object]) -> pd.DataFrame:
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
                "日期": parts[0],
                "开盘": float(parts[1]),
                "收盘": float(parts[2]),
                "最高": float(parts[3]),
                "最低": float(parts[4]),
                "成交量": float(parts[5]),
                "成交额": float(parts[6]),
            }
        )
    if not rows:
        raise DataProviderError("Eastmoney history fallback returned no parseable kline rows")
    return pd.DataFrame(rows)


class AkShareProvider:
    data_mode = DATA_MODE_REAL_PUBLIC

    def __init__(self, akshare_client: object | None = None, adjustment: str = "qfq") -> None:
        if adjustment != "qfq":
            raise ValueError(
                "AkShareProvider adjustment must be 'qfq' for v0.2 real-data ingestion"
            )
        if akshare_client is None:
            try:
                import akshare as akshare_module
            except ImportError as exc:
                raise DataProviderError(
                    "AkShare is optional and required only for --provider akshare. "
                    "Install akshare to use real public market data."
                ) from exc
            try:
                import requests
            except ImportError:
                requests = None
            http_get = requests.get if requests is not None else None
            akshare_client = _DefaultAkShareClient(akshare_module, http_get=http_get)
        self._client = akshare_client
        self.adjustment = adjustment

    def get_stock_info(self) -> pd.DataFrame:
        return self._call_and_normalize("stock_info", self._client.stock_info, normalize_stock_info)

    def get_price_history(self, codes: list[str]) -> pd.DataFrame:
        frames = [
            self._call_and_normalize(
                "stock_daily",
                lambda code=code: self._client.stock_daily(code, self.adjustment),
                lambda raw, code=code: normalize_price_history(raw, code=code),
            )
            for code in codes
        ]
        if not frames:
            return normalize_price_history(pd.DataFrame())
        return pd.concat(frames, ignore_index=True)

    def get_index_history(self) -> pd.DataFrame:
        return pd.concat(
            [
                self._call_and_normalize(
                    "index_daily",
                    lambda code=code: self._client.index_daily(code),
                    lambda raw, code=code: normalize_index_history(raw, code=code),
                )
                for code in ("000300", "000001.SH", "399006.SZ")
            ],
            ignore_index=True,
        )

    @staticmethod
    def _call_and_normalize(
        operation: str,
        fetch: Callable[[], pd.DataFrame],
        normalize: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        try:
            return normalize(fetch())
        except Exception as exc:
            raise DataProviderError(f"AkShare {operation} failed: {exc}") from exc


_LIST_COLUMNS = {
    "code": ("code", "\u4ee3\u7801", "\u8bc1\u5238\u4ee3\u7801", "A\u80a1\u4ee3\u7801"),
    "name": ("name", "\u540d\u79f0", "\u8bc1\u5238\u7b80\u79f0", "A\u80a1\u7b80\u79f0"),
}
_SPOT_COLUMNS = {
    **_LIST_COLUMNS,
    "latest_price": ("latest_price", "\u6700\u65b0\u4ef7"),
    "spot_volume": ("spot_volume", "\u6210\u4ea4\u91cf"),
    "spot_amount": ("spot_amount", "\u6210\u4ea4\u989d"),
    "explicit_suspension": ("is_suspended", "\u662f\u5426\u505c\u724c", "\u505c\u724c"),
    "avg_turnover_20d": ("avg_turnover_20d", "20\u65e5\u5e73\u5747\u6210\u4ea4\u989d"),
}
_SH_LISTING_COLUMNS = {
    "code": ("\u8bc1\u5238\u4ee3\u7801", "code"),
    "listing_date": ("\u4e0a\u5e02\u65e5\u671f", "listing_date"),
}
_SZ_LISTING_COLUMNS = {
    "code": ("A\u80a1\u4ee3\u7801", "code"),
    "listing_date": ("A\u80a1\u4e0a\u5e02\u65e5\u671f", "listing_date"),
}
_LISTING_SOURCES = (
    ("stock_info_sh_name_code", {"symbol": "\u4e3b\u677fA\u80a1"}, _SH_LISTING_COLUMNS),
    ("stock_info_sh_name_code", {"symbol": "\u79d1\u521b\u677f"}, _SH_LISTING_COLUMNS),
    ("stock_info_sz_name_code", {"symbol": "A\u80a1\u5217\u8868"}, _SZ_LISTING_COLUMNS),
)


def _build_base_stock_info(module: object) -> pd.DataFrame:
    list_raw = _optional_source(module, "stock_info_a_code_name")
    base = _select_columns(list_raw, _LIST_COLUMNS)
    if not {"code", "name"}.issubset(base.columns):
        raise DataProviderError(
            "AkShare stock list source is insufficient for code/name"
        )

    base = base.loc[:, ["code", "name"]].copy()
    base["code"] = base["code"].map(_normalize_code)
    if base["code"].eq("").any():
        raise DataProviderError("AkShare stock list source is insufficient for code/name")
    base = base.drop_duplicates("code")
    if base.empty:
        raise DataProviderError("AkShare stock list source is insufficient for code/name")
    if base["name"].isna().any() or base["name"].astype(str).str.strip().eq("").any():
        raise DataProviderError("AkShare stock list source is insufficient for code/name")

    listing_frames: list[pd.DataFrame] = []
    for method_name, kwargs, columns in _LISTING_SOURCES:
        raw = _optional_source(module, method_name, **kwargs)
        standardized = _select_columns(raw, columns)
        if {"code", "listing_date"}.issubset(standardized.columns):
            standardized["code"] = standardized["code"].map(_normalize_code)
            listing_frames.append(standardized.loc[:, ["code", "listing_date"]])

    if listing_frames:
        listing = pd.concat(listing_frames, ignore_index=True).drop_duplicates("code")
        base = base.merge(listing, on="code", how="left")
    else:
        base["listing_date"] = None

    base["listing_date"] = base["listing_date"].map(_parse_listing_date)
    missing_listing = base["listing_date"].isna() & base["code"].map(
        _requires_listing_date
    )
    for index in base.index[missing_listing]:
        detail = _optional_source(
            module,
            "stock_individual_info_em",
            symbol=base.at[index, "code"],
        )
        listing_date = _individual_listing_date(detail)
        if listing_date is not None:
            base.at[index, "listing_date"] = listing_date

    missing_codes = base.loc[
        base["listing_date"].isna() & base["code"].map(_requires_listing_date),
        "code",
    ].tolist()
    if missing_codes:
        preview = ", ".join(missing_codes[:5])
        raise DataProviderError(
            "AkShare metadata sources are insufficient for listing_date: " + preview
        )

    base["is_suspended"] = False
    base["avg_turnover_20d"] = float("nan")
    return base


def _enrich_with_optional_spot(
    base: pd.DataFrame,
    spot_raw: pd.DataFrame | None,
) -> pd.DataFrame:
    spot = _select_columns(spot_raw, _SPOT_COLUMNS)
    if "code" not in spot:
        return base.copy()

    spot = spot.copy()
    spot["code"] = spot["code"].map(_normalize_code)
    spot = spot[spot["code"].ne("")].drop_duplicates("code")
    merged = base.merge(
        spot.drop(columns=["name"], errors="ignore"), on="code", how="left"
    )

    explicit = merged.get("explicit_suspension")
    if explicit is not None:
        explicit_values = explicit.map(_to_bool)
        merged["is_suspended"] = explicit_values.where(
            explicit.notna(), merged["is_suspended"]
        )
    else:
        latest = _numeric_column(merged, "latest_price")
        volume = _numeric_column(merged, "spot_volume")
        amount = _numeric_column(merged, "spot_amount")
        inferred = latest.isna() & volume.fillna(0).eq(0) & amount.fillna(0).eq(0)
        has_spot_metrics = latest.notna() | volume.notna() | amount.notna()
        merged["is_suspended"] = inferred.where(
            has_spot_metrics, merged["is_suspended"]
        )

    if "avg_turnover_20d_y" in merged.columns:
        merged["avg_turnover_20d"] = merged["avg_turnover_20d_y"].where(
            merged["avg_turnover_20d_y"].notna(),
            merged.get("avg_turnover_20d_x"),
        )
        merged = merged.drop(columns=["avg_turnover_20d_x", "avg_turnover_20d_y"])

    if "avg_turnover_20d" not in merged:
        merged["avg_turnover_20d"] = float("nan")
    return merged

def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) < 6 else text


def _select_columns(
    frame: pd.DataFrame | None,
    mapping: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    result = pd.DataFrame(index=frame.index)
    for target, candidates in mapping.items():
        source = next((column for column in candidates if column in frame.columns), None)
        if source is not None:
            result[target] = frame[source]
    return result


def _optional_source(module: object, method_name: str, **kwargs: object) -> pd.DataFrame | None:
    method = getattr(module, method_name, None)
    if method is None:
        return None
    try:
        frame = method(**kwargs)
    except Exception:
        return None
    return frame if isinstance(frame, pd.DataFrame) else None


def _parse_listing_date(value: object) -> object | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) == 8:
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _requires_listing_date(code: str) -> bool:
    return str(code).startswith(
        ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")
    )


def _individual_listing_date(frame: pd.DataFrame | None) -> object | None:
    if frame is None or not {"item", "value"}.issubset(frame.columns):
        return None
    values = dict(zip(frame["item"].astype(str), frame["value"], strict=False))
    for key in ("\u4e0a\u5e02\u65f6\u95f4", "\u4e0a\u5e02\u65e5\u671f"):
        parsed = _parse_listing_date(values.get(key))
        if parsed is not None:
            return parsed
    return None


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "\u662f", "\u505c\u724c"}


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.get(column, pd.Series(index=frame.index, dtype=float))
    return pd.to_numeric(values, errors="coerce")
