from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from a_share_ai.models import (
    AppConfig,
    BacktestConfig,
    ProviderConfig,
    ReportConfig,
    RiskConfig,
    StrategyConfig,
    UniverseConfig,
)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return AppConfig(
        universe=UniverseConfig(**_section(raw, "universe")),
        strategy=StrategyConfig(**_section(raw, "strategy")),
        risk=RiskConfig(**_section(raw, "risk")),
        reports=ReportConfig(**_section(raw, "reports")),
        backtest=BacktestConfig(**_section(raw, "backtest")),
        provider=ProviderConfig(**_section(raw, "provider")),
    )
