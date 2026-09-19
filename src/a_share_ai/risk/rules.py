from __future__ import annotations

from dataclasses import replace

import pandas as pd

from a_share_ai.models import RiskConfig, RiskState


def _risk_decision(config: RiskConfig, state: RiskState) -> tuple[bool, str, str, float]:
    if state.current_drawdown <= config.portfolio_reduce_risk_drawdown:
        return False, "observe_only", "组合回撤超过-15%，进入只观察/不新增买入模式", 0.0
    if state.current_holdings >= config.max_focused_holdings:
        return False, "hold_limit", "持仓数量达到上限", 0.0
    if state.daily_new_buys >= config.max_daily_new_buys:
        return False, "daily_buy_limit", "单日新增买入达到上限", 0.0
    if state.current_drawdown <= config.portfolio_warning_drawdown:
        return True, "warning", "组合回撤超过-12%，允许关注但需要降低出手频率", config.single_position_cap / 2
    return True, "normal", "风险状态正常", config.single_position_cap


def apply_risk_rules(
    candidates: pd.DataFrame,
    config: RiskConfig,
    state: RiskState,
) -> pd.DataFrame:
    data = candidates.copy()
    working_state = state
    decisions: list[tuple[bool, str, str, float]] = []

    for _ in data.index:
        decision = _risk_decision(config, working_state)
        decisions.append(decision)
        if decision[0]:
            working_state = replace(
                working_state,
                current_holdings=working_state.current_holdings + 1,
                daily_new_buys=working_state.daily_new_buys + 1,
            )

    data["can_buy"] = pd.Series(
        [decision[0] for decision in decisions], index=data.index, dtype=object
    )
    data["risk_action"] = [decision[1] for decision in decisions]
    data["risk_notes"] = [decision[2] for decision in decisions]
    data["suggested_position"] = [decision[3] for decision in decisions]
    return data
