from __future__ import annotations

import pandas as pd

from a_share_ai.models import StrategyConfig


def rank_candidates(scored_rows: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = scored_rows.copy()
    if "risk_penalty" not in data.columns:
        data["risk_penalty"] = 0.0

    data["total_score"] = (
        data["medium_term_score"] * config.medium_term_weight
        + data["timing_score"] * config.timing_weight
        - data["risk_penalty"] * config.risk_weight
    ).round(2)
    data = data.sort_values(["total_score", "medium_term_score"], ascending=False).reset_index(drop=True)
    data["tier"] = "excluded"
    candidate_limit = min(config.candidate_count, len(data))
    focused_limit = min(config.focused_count, candidate_limit)
    data.loc[: candidate_limit - 1, "tier"] = "candidate"
    data.loc[: focused_limit - 1, "tier"] = "focused"
    return data
