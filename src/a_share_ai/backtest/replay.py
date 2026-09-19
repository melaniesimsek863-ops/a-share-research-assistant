from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReplayConfig:
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)


def evaluate_forward_returns(
    candidate_snapshots: pd.DataFrame,
    price_history: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    horizons_tuple = tuple(horizons)
    if candidate_snapshots.empty:
        return _empty_evaluated(candidate_snapshots, horizons_tuple)

    required_candidate_columns = {"date", "code"}
    required_price_columns = {"date", "code", "close"}
    _require_columns(candidate_snapshots, required_candidate_columns, "candidate_snapshots")
    _require_columns(price_history, required_price_columns, "price_history")

    prices = price_history.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["code", "date"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _, candidate in candidate_snapshots.iterrows():
        row = candidate.to_dict()
        candidate_date = pd.Timestamp(candidate["date"])
        code = candidate["code"]
        code_prices = prices[(prices["code"] == code) & (prices["date"] >= candidate_date)].reset_index(drop=True)

        if code_prices.empty:
            row["entry_date"] = pd.NaT
            row["entry_close"] = float("nan")
            for horizon in horizons_tuple:
                row[f"forward_date_{horizon}d"] = pd.NaT
                row[f"forward_return_{horizon}d"] = float("nan")
                row[f"has_forward_{horizon}d"] = False
            rows.append(row)
            continue

        entry = code_prices.iloc[0]
        entry_close = float(entry["close"])
        row["entry_date"] = entry["date"]
        row["entry_close"] = entry_close

        for horizon in horizons_tuple:
            if len(code_prices) > horizon:
                forward = code_prices.iloc[horizon]
                row[f"forward_date_{horizon}d"] = forward["date"]
                row[f"forward_return_{horizon}d"] = round(float(forward["close"]) / entry_close - 1, 4)
                row[f"has_forward_{horizon}d"] = True
            else:
                row[f"forward_date_{horizon}d"] = pd.NaT
                row[f"forward_return_{horizon}d"] = float("nan")
                row[f"has_forward_{horizon}d"] = False
        rows.append(row)

    evaluated = pd.DataFrame(rows)
    for horizon in horizons_tuple:
        evaluated[f"has_forward_{horizon}d"] = evaluated[f"has_forward_{horizon}d"].astype(object)
    return evaluated


def summarize_replay_results(
    evaluated_candidates: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for horizon in tuple(horizons):
        return_column = f"forward_return_{horizon}d"
        has_column = f"has_forward_{horizon}d"
        if evaluated_candidates.empty or return_column not in evaluated_candidates or has_column not in evaluated_candidates:
            valid_returns = pd.Series(dtype=float)
        else:
            valid_returns = evaluated_candidates.loc[
                evaluated_candidates[has_column].fillna(False).astype(bool),
                return_column,
            ].dropna().astype(float)

        if valid_returns.empty:
            rows.append(
                {
                    "horizon": horizon,
                    "sample_count": 0,
                    "avg_return": 0.0,
                    "median_return": 0.0,
                    "win_rate": 0.0,
                }
            )
            continue

        rows.append(
            {
                "horizon": horizon,
                "sample_count": len(valid_returns),
                "avg_return": round(float(valid_returns.mean()), 4),
                "median_return": round(float(valid_returns.median()), 4),
                "win_rate": round(float((valid_returns > 0).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def run_historical_replay(
    candidate_snapshots: pd.DataFrame,
    price_history: pd.DataFrame,
    config: ReplayConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_config = config or ReplayConfig()
    evaluated = evaluate_forward_returns(candidate_snapshots, price_history, replay_config.horizons)
    summary = summarize_replay_results(evaluated, replay_config.horizons)
    return evaluated, summary


def _empty_evaluated(candidate_snapshots: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    evaluated = candidate_snapshots.copy()
    for horizon in horizons:
        evaluated[f"forward_date_{horizon}d"] = pd.Series(dtype="datetime64[ns]")
        evaluated[f"forward_return_{horizon}d"] = pd.Series(dtype=float)
        evaluated[f"has_forward_{horizon}d"] = pd.Series(dtype=object)
    return evaluated


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
