from __future__ import annotations

import re

import pandas as pd

from a_share_ai.models import StrategyConfig

EXPERIMENTAL_COLUMNS = [
    "experimental_medium_term_score",
    "experimental_timing_score",
    "experimental_total_score",
    "experimental_rank",
    "experimental_tier",
]

EXPERIMENTAL_V2_REQUIRED_COLUMNS = [
    "signal_labels",
    "above_ma20",
    "ma_bullish",
    "return_20d",
    "return_60d",
    "drawdown_60d",
    "volume_ratio_5_20",
]

EXPERIMENTAL_V2_COLUMNS = [
    "experimental_v2_quality_score",
    "experimental_v2_momentum_score",
    "experimental_v2_timing_score",
    "experimental_v2_penalty_score",
    "experimental_v2_total_score",
    "experimental_v2_rank",
    "experimental_v2_tier",
]

EXPERIMENTAL_V3_REQUIRED_COLUMNS = EXPERIMENTAL_V2_REQUIRED_COLUMNS

EXPERIMENTAL_V3_COLUMNS = [
    "experimental_v3_quality_score",
    "experimental_v3_timing_score",
    "experimental_v3_anti_overheat_score",
    "experimental_v3_overheat_penalty_score",
    "experimental_v3_total_score",
    "experimental_v3_rank",
    "experimental_v3_tier",
]


def score_experimental_layer(evaluated: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    original_index = evaluated.index.copy()
    data = evaluated.copy().reset_index(drop=True)
    data["experimental_medium_term_score"] = _experimental_medium_term_score(data)
    data["experimental_timing_score"] = _experimental_timing_score(data)
    risk_penalty = data.get("risk_penalty", pd.Series(0.0, index=data.index))
    data["experimental_total_score"] = (
        data["experimental_medium_term_score"] * config.medium_term_weight
        + data["experimental_timing_score"] * config.timing_weight
        - pd.to_numeric(risk_penalty, errors="coerce").fillna(0.0) * config.risk_weight
    ).clip(0, 100).round(2)
    scored = _assign_experimental_tiers(data, config)
    scored.index = original_index
    return scored


def score_experimental_v2_layer(evaluated: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    _validate_experimental_v2_columns(evaluated)
    original_index = evaluated.index.copy()
    data = evaluated.copy().reset_index(drop=True)
    data["experimental_v2_quality_score"] = _experimental_v2_quality_score(data)
    data["experimental_v2_momentum_score"] = _experimental_v2_momentum_score(data)
    data["experimental_v2_timing_score"] = _experimental_v2_timing_score(data)
    data["experimental_v2_penalty_score"] = _experimental_v2_penalty_score(data)
    data["experimental_v2_total_score"] = (
        data["experimental_v2_quality_score"] * 0.40
        + data["experimental_v2_momentum_score"] * 0.30
        + data["experimental_v2_timing_score"] * 0.20
        - data["experimental_v2_penalty_score"] * 0.30
    ).clip(0, 100).round(2)
    scored = _assign_experimental_v2_tiers(data, config)
    scored.index = original_index
    return scored


def score_experimental_v3_layer(evaluated: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    _validate_experimental_v3_columns(evaluated)
    original_index = evaluated.index.copy()
    data = evaluated.copy().reset_index(drop=True)
    data["experimental_v3_quality_score"] = _experimental_v3_quality_score(data)
    data["experimental_v3_timing_score"] = _experimental_v3_timing_score(data)
    data["experimental_v3_anti_overheat_score"] = _experimental_v3_anti_overheat_score(data)
    data["experimental_v3_overheat_penalty_score"] = _experimental_v3_overheat_penalty_score(data)
    data["experimental_v3_total_score"] = (
        data["experimental_v3_quality_score"] * 0.45
        + data["experimental_v3_timing_score"] * 0.25
        + data["experimental_v3_anti_overheat_score"] * 0.20
        - data["experimental_v3_overheat_penalty_score"] * 0.35
    ).clip(0, 100).round(2)
    scored = _assign_experimental_v3_tiers(data, config)
    scored.index = original_index
    return scored


def _validate_experimental_v2_columns(evaluated: pd.DataFrame) -> None:
    missing = [column for column in EXPERIMENTAL_V2_REQUIRED_COLUMNS if column not in evaluated.columns]
    if missing:
        raise ValueError(f"Missing required experimental v2 scoring columns: {missing}")


def _validate_experimental_v3_columns(evaluated: pd.DataFrame) -> None:
    missing = [column for column in EXPERIMENTAL_V3_REQUIRED_COLUMNS if column not in evaluated.columns]
    if missing:
        raise ValueError(f"Missing required experimental v3 scoring columns: {missing}")


def _experimental_medium_term_score(data: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=data.index)
    score += data["above_ma20"].astype(float) * 15
    score += data["ma_bullish"].astype(float) * 20
    score += _triangular_score(data["return_20d"], low=-0.05, peak=0.08, high=0.22) * 25
    score += _triangular_score(data["return_60d"], low=-0.10, peak=0.18, high=0.42) * 25
    score += _triangular_score(data["drawdown_60d"], low=-0.20, peak=-0.07, high=0.0) * 15
    overheat = (
        (pd.to_numeric(data["return_20d"], errors="coerce") > 0.25)
        | (pd.to_numeric(data["return_60d"], errors="coerce") > 0.50)
    )
    score -= overheat.astype(float) * 20
    return score.clip(0, 100).round(2)


def _experimental_v2_quality_score(data: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=data.index)
    score += _boolean_column(data, "above_ma20").astype(float) * 22
    score += _boolean_column(data, "ma_bullish").astype(float) * 28
    score += _triangular_score(_numeric_column(data, "drawdown_60d"), low=-0.18, peak=-0.07, high=-0.01) * 30
    score += _triangular_score(_numeric_column(data, "return_60d"), low=-0.02, peak=0.18, high=0.38) * 20
    return score.clip(0, 100).round(2)


def _experimental_v2_momentum_score(data: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=data.index)
    score += _triangular_score(_numeric_column(data, "return_20d"), low=-0.03, peak=0.08, high=0.20) * 45
    score += _triangular_score(_numeric_column(data, "return_60d"), low=0.00, peak=0.18, high=0.42) * 45
    score += _triangular_score(_numeric_column(data, "volume_ratio_5_20"), low=0.70, peak=1.50, high=3.00) * 10
    return score.clip(0, 100).round(2)


def _experimental_v2_timing_score(data: pd.DataFrame) -> pd.Series:
    values = []
    for _, row in data.iterrows():
        labels = _split_signal_labels(row.get("signal_labels", ""))
        score = 0.0
        if "放量且收盘在20日均线上方" in labels:
            score += 35
        if "收盘在20日均线上方且60日回撤大于-8%" in labels:
            score += 25
        if "20日涨幅低于18%" in labels:
            score += 25
        if row.get("above_ma20", False) and row.get("ma_bullish", False):
            score += 15
        values.append(score)
    return pd.Series(values, index=data.index, dtype=float).clip(0, 100).round(2)


def _experimental_v2_penalty_score(data: pd.DataFrame) -> pd.Series:
    return_20d = _numeric_column(data, "return_20d")
    return_60d = _numeric_column(data, "return_60d")
    drawdown = _numeric_column(data, "drawdown_60d")
    score = pd.Series(0.0, index=data.index)
    score += (return_20d > 0.22).astype(float) * 35
    score += (return_60d > 0.45).astype(float) * 35
    score += ((drawdown > -0.03) & ((return_20d > 0.18) | (return_60d > 0.35))).astype(float) * 20
    score += data.reindex(columns=["return_20d", "return_60d", "drawdown_60d"]).isna().sum(axis=1) * 10
    return score.clip(0, 100).round(2)


def _experimental_v3_quality_score(data: pd.DataFrame) -> pd.Series:
    return_20d = _numeric_column(data, "return_20d")
    return_60d = _numeric_column(data, "return_60d")
    drawdown = _numeric_column(data, "drawdown_60d")
    score = pd.Series(0.0, index=data.index)
    score += _boolean_column(data, "above_ma20").astype(float) * 16
    score += _boolean_column(data, "ma_bullish").astype(float) * 12
    score += _triangular_score(return_20d, low=-0.04, peak=0.08, high=0.18) * 26
    score += _triangular_score(return_60d, low=-0.04, peak=0.16, high=0.34) * 26
    score += _triangular_score(drawdown, low=-0.18, peak=-0.08, high=-0.03) * 20
    return score.clip(0, 100).round(2)


def _experimental_v3_timing_score(data: pd.DataFrame) -> pd.Series:
    values = []
    for _, row in data.iterrows():
        labels = _split_signal_labels(row.get("signal_labels", ""))
        score = 0.0
        if "20日涨幅低于18%" in labels:
            score += 35
        if "放量且收盘在20日均线上方" in labels:
            score += 25
        if "收盘在20日均线上方且60日回撤大于-8%" in labels:
            score += 25
        if row.get("above_ma20", False) and not row.get("ma_bullish", False):
            score += 5
        values.append(score)
    return pd.Series(values, index=data.index, dtype=float).clip(0, 100).round(2)


def _experimental_v3_anti_overheat_score(data: pd.DataFrame) -> pd.Series:
    return_20d = _numeric_column(data, "return_20d")
    return_60d = _numeric_column(data, "return_60d")
    volume_ratio = _numeric_column(data, "volume_ratio_5_20")
    score = pd.Series(0.0, index=data.index)
    score += return_20d.between(0.00, 0.18, inclusive="both").astype(float) * 35
    score += return_60d.between(0.00, 0.35, inclusive="both").astype(float) * 35
    score += volume_ratio.between(0.80, 2.20, inclusive="both").astype(float) * 20
    score += return_20d.between(0.00, 0.18, inclusive="both").astype(float) * 10
    return score.clip(0, 100).round(2)


def _experimental_v3_overheat_penalty_score(data: pd.DataFrame) -> pd.Series:
    return_20d = _numeric_column(data, "return_20d")
    return_60d = _numeric_column(data, "return_60d")
    drawdown = _numeric_column(data, "drawdown_60d")
    volume_ratio = _numeric_column(data, "volume_ratio_5_20")
    score = pd.Series(0.0, index=data.index)
    score += (return_20d > 0.18).astype(float) * 18
    score += (return_20d > 0.25).astype(float) * 18
    score += (return_60d > 0.35).astype(float) * 18
    score += (return_60d > 0.45).astype(float) * 18
    score += ((drawdown > -0.03) & ((return_20d > 0.18) | (return_60d > 0.35))).astype(float) * 18
    score += (volume_ratio > 3.0).astype(float) * 10
    score += data.reindex(columns=["return_20d", "return_60d", "drawdown_60d", "volume_ratio_5_20"]).isna().sum(axis=1) * 10
    return score.clip(0, 100).round(2)


def _experimental_timing_score(data: pd.DataFrame) -> pd.Series:
    values = []
    for _, row in data.iterrows():
        labels = _split_signal_labels(row.get("signal_labels", ""))
        score = 0.0
        if "放量且收盘在20日均线上方" in labels:
            score += 35
        if "收盘在20日均线上方且60日回撤大于-8%" in labels:
            score += 15
        if "20日涨幅低于18%" in labels:
            score += 20
        return_20d = pd.to_numeric(pd.Series([row.get("return_20d")]), errors="coerce").iloc[0]
        return_60d = pd.to_numeric(pd.Series([row.get("return_60d")]), errors="coerce").iloc[0]
        if return_20d > 0.25:
            score -= 25
        if return_60d > 0.50:
            score -= 25
        values.append(max(0.0, min(100.0, score)))
    return pd.Series(values, index=data.index, dtype=float).round(2)


def _assign_experimental_tiers(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    assigned = data.copy()
    assigned["experimental_rank"] = 0
    assigned["experimental_tier"] = "excluded"
    date_groups = (
        assigned.groupby("date", dropna=False, sort=False)
        if "date" in assigned.columns
        else [(None, assigned)]
    )
    for _, group in date_groups:
        ranked = group.sort_values(
            ["experimental_total_score", "experimental_medium_term_score"],
            ascending=False,
            kind="mergesort",
        )
        candidate_limit = min(config.candidate_count, len(ranked))
        focused_limit = min(config.focused_count, candidate_limit)
        assigned.loc[ranked.index, "experimental_rank"] = range(1, len(ranked) + 1)
        assigned.loc[ranked.index[:candidate_limit], "experimental_tier"] = "candidate"
        assigned.loc[ranked.index[:focused_limit], "experimental_tier"] = "focused"
    return assigned


def _assign_experimental_v2_tiers(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    assigned = data.copy()
    assigned["experimental_v2_rank"] = 0
    assigned["experimental_v2_tier"] = "excluded"
    date_groups = (
        assigned.groupby("date", dropna=False, sort=False)
        if "date" in assigned.columns
        else [(None, assigned)]
    )
    for _, group in date_groups:
        ranked = group.sort_values(
            ["experimental_v2_total_score", "experimental_v2_quality_score"],
            ascending=False,
            kind="mergesort",
        )
        candidate_limit = min(config.candidate_count, len(ranked))
        focused_limit = min(config.focused_count, candidate_limit)
        assigned.loc[ranked.index, "experimental_v2_rank"] = range(1, len(ranked) + 1)
        assigned.loc[ranked.index[:candidate_limit], "experimental_v2_tier"] = "candidate"
        assigned.loc[ranked.index[:focused_limit], "experimental_v2_tier"] = "focused"
    return assigned


def _assign_experimental_v3_tiers(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    assigned = data.copy()
    assigned["experimental_v3_rank"] = 0
    assigned["experimental_v3_tier"] = "excluded"
    date_groups = (
        assigned.groupby("date", dropna=False, sort=False)
        if "date" in assigned.columns
        else [(None, assigned)]
    )
    for _, group in date_groups:
        ranked = group.sort_values(
            ["experimental_v3_total_score", "experimental_v3_quality_score"],
            ascending=False,
            kind="mergesort",
        )
        candidate_limit = min(config.candidate_count, len(ranked))
        focused_limit = min(config.focused_count, candidate_limit)
        assigned.loc[ranked.index, "experimental_v3_rank"] = range(1, len(ranked) + 1)
        assigned.loc[ranked.index[:candidate_limit], "experimental_v3_tier"] = "candidate"
        assigned.loc[ranked.index[:focused_limit], "experimental_v3_tier"] = "focused"
    return assigned


def _triangular_score(values: pd.Series, *, low: float, peak: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(low)
    left = ((numeric - low) / (peak - low)).clip(lower=0, upper=1)
    right = ((high - numeric) / (high - peak)).clip(lower=0, upper=1)
    return pd.concat([left, right], axis=1).min(axis=1).clip(lower=0, upper=1)


def _numeric_column(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def _boolean_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(False, index=data.index)
    return data[column].fillna(False).astype(bool)


def _split_signal_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text.strip("[]").replace("'", "").replace('"', "")
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
