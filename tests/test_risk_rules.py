import pandas as pd

from a_share_ai.config import load_config
from a_share_ai.models import RiskState
from a_share_ai.risk.rules import apply_risk_rules


def test_risk_rules_cap_position_and_allow_buy_when_drawdown_ok():
    config = load_config("configs/default.yaml")
    candidates = pd.DataFrame([{"code": "000001", "tier": "focused", "total_score": 88.0}])
    result = apply_risk_rules(
        candidates,
        config.risk,
        RiskState(current_drawdown=-0.05, current_holdings=2, daily_new_buys=1),
    )

    assert result.iloc[0]["suggested_position"] == 0.15
    assert result.iloc[0]["can_buy"] is True
    assert result.iloc[0]["risk_action"] == "normal"


def test_risk_rules_block_new_buy_when_portfolio_drawdown_too_large():
    config = load_config("configs/default.yaml")
    candidates = pd.DataFrame([{"code": "000001", "tier": "focused", "total_score": 88.0}])
    result = apply_risk_rules(
        candidates,
        config.risk,
        RiskState(current_drawdown=-0.16, current_holdings=2, daily_new_buys=1),
    )

    assert result.iloc[0]["suggested_position"] == 0.0
    assert result.iloc[0]["can_buy"] is False
    assert result.iloc[0]["risk_action"] == "observe_only"
    assert "组合回撤超过-15%" in result.iloc[0]["risk_notes"]


def test_risk_rules_warning_at_portfolio_drawdown_boundary():
    config = load_config("configs/default.yaml")
    candidates = pd.DataFrame([{"code": "000001", "tier": "focused", "total_score": 88.0}])
    result = apply_risk_rules(
        candidates,
        config.risk,
        RiskState(current_drawdown=-0.12, current_holdings=2, daily_new_buys=1),
    )

    assert result.iloc[0]["suggested_position"] == 0.075
    assert result.iloc[0]["can_buy"] is True
    assert result.iloc[0]["risk_action"] == "warning"


def test_risk_rules_block_new_buy_at_max_holdings_boundary():
    config = load_config("configs/default.yaml")
    candidates = pd.DataFrame([{"code": "000001", "tier": "focused", "total_score": 88.0}])
    result = apply_risk_rules(
        candidates,
        config.risk,
        RiskState(current_drawdown=-0.05, current_holdings=6, daily_new_buys=1),
    )

    assert result.iloc[0]["suggested_position"] == 0.0
    assert result.iloc[0]["can_buy"] is False
    assert result.iloc[0]["risk_action"] == "hold_limit"


def test_risk_rules_block_new_buy_at_daily_buy_cap_boundary():
    config = load_config("configs/default.yaml")
    candidates = pd.DataFrame([{"code": "000001", "tier": "focused", "total_score": 88.0}])
    result = apply_risk_rules(
        candidates,
        config.risk,
        RiskState(current_drawdown=-0.05, current_holdings=2, daily_new_buys=3),
    )

    assert result.iloc[0]["suggested_position"] == 0.0


def test_risk_rules_consume_daily_buy_and_holding_capacity_in_rank_order():
    config = load_config("configs/default.yaml")
    candidates = pd.DataFrame(
        [
            {"code": "000001", "tier": "focused", "total_score": 91.0},
            {"code": "000002", "tier": "focused", "total_score": 89.0},
            {"code": "000003", "tier": "focused", "total_score": 87.0},
            {"code": "000004", "tier": "candidate", "total_score": 85.0},
        ]
    )

    result = apply_risk_rules(
        candidates,
        config.risk,
        RiskState(current_drawdown=-0.05, current_holdings=4, daily_new_buys=1),
    )

    assert result["can_buy"].tolist() == [True, True, False, False]
    assert result["suggested_position"].tolist() == [0.15, 0.15, 0.0, 0.0]
    assert result["risk_action"].tolist() == [
        "normal",
        "normal",
        "hold_limit",
        "hold_limit",
    ]
