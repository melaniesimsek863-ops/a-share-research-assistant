from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from a_share_ai.data.schema import REQUIRED_BAR_COLUMNS, _normalize_code, normalize_price_history


MANIFEST_COLUMNS = (
    "code",
    "latest_cached_date",
    "last_attempt_at",
    "last_success_at",
    "status",
    "failure_reason",
    "consecutive_failures",
)


def ensure_cache_dir(base_dir: str | Path = "data/raw") -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(dataset: str, trade_date: str, base_dir: str | Path = "data/raw") -> Path:
    return ensure_cache_dir(base_dir) / f"{dataset}_{trade_date}.csv"


class CsvMarketCache:
    def __init__(self, base_dir: str | Path = "data/raw/akshare") -> None:
        self.base_dir = Path(base_dir)
        self.daily_dir = self.base_dir / "daily"
        self.manifest_path = self.base_dir / "manifest.csv"

    def read_daily(self, code: str) -> pd.DataFrame:
        path = self.daily_dir / f"{code}.csv"
        if not path.exists():
            return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS)
        raw = pd.read_csv(path)
        return normalize_price_history(raw, code=code)

    def write_daily(self, code: str, bars: pd.DataFrame) -> None:
        normalized = normalize_price_history(bars, code=code)
        normalized = (
            normalized.drop_duplicates(["date", "code"], keep="last")
            .sort_values(["date", "code"])
            .reset_index(drop=True)
        )
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(self.daily_dir / f"{code}.csv", index=False)

    def read_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        data = pd.read_csv(self.manifest_path)
        data["code"] = data["code"].map(_normalize_code)
        for column in MANIFEST_COLUMNS:
            if column not in data.columns:
                data[column] = None
        for column in ("last_attempt_at", "last_success_at", "status", "failure_reason"):
            data[column] = data[column].fillna("").astype(str)
        data["latest_cached_date"] = pd.to_datetime(
            data["latest_cached_date"], errors="coerce"
        ).dt.date.astype(object)
        data["consecutive_failures"] = (
            pd.to_numeric(data["consecutive_failures"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        return data.loc[:, MANIFEST_COLUMNS]

    def update_manifest_success(self, code: str, latest_date: date | None) -> None:
        code = _normalize_code(code)
        manifest = self._upsert_manifest(code)
        now = _now_iso()
        row_index = manifest.index[manifest["code"].eq(code)][0]
        manifest.at[row_index, "latest_cached_date"] = latest_date
        manifest.at[row_index, "last_attempt_at"] = now
        manifest.at[row_index, "last_success_at"] = now
        manifest.at[row_index, "status"] = "ok"
        manifest.at[row_index, "failure_reason"] = ""
        manifest.at[row_index, "consecutive_failures"] = 0
        self._write_manifest(manifest)

    def update_manifest_failure(self, code: str, reason: str) -> None:
        code = _normalize_code(code)
        manifest = self._upsert_manifest(code)
        row_index = manifest.index[manifest["code"].eq(code)][0]
        current_failures = int(manifest.at[row_index, "consecutive_failures"] or 0)
        manifest.at[row_index, "last_attempt_at"] = _now_iso()
        manifest.at[row_index, "status"] = "failed"
        manifest.at[row_index, "failure_reason"] = str(reason)
        manifest.at[row_index, "consecutive_failures"] = current_failures + 1
        self._write_manifest(manifest)

    def _upsert_manifest(self, code: str) -> pd.DataFrame:
        manifest = self.read_manifest()
        if not manifest["code"].astype(str).eq(code).any():
            manifest = pd.concat(
                [
                    manifest,
                    pd.DataFrame(
                        [{
                            "code": code,
                            "latest_cached_date": None,
                            "last_attempt_at": "",
                            "last_success_at": "",
                            "status": "unknown",
                            "failure_reason": "",
                            "consecutive_failures": 0,
                        }]
                    ),
                ],
                ignore_index=True,
            )
        return manifest

    def _write_manifest(self, manifest: pd.DataFrame) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        manifest.loc[:, MANIFEST_COLUMNS].to_csv(self.manifest_path, index=False)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
