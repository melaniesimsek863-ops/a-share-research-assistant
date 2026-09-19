from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


BOARD_MAIN = "main"
BOARD_CHINEXT = "chinext"
BOARD_STAR = "star"

DATA_MODE_SYNTHETIC_DEMO = "synthetic_demo"
DATA_MODE_REAL_PUBLIC = "real_public_market_data"

QUALITY_OK = "ok"
QUALITY_WARNING = "warning"
QUALITY_BLOCKED = "blocked"


@dataclass(frozen=True)
class UniverseConfig:
    include_boards: list[str] = field(default_factory=lambda: [BOARD_MAIN, BOARD_CHINEXT])
    exclude_star_market: bool = True
    exclude_st: bool = True
    exclude_delisting_risk: bool = True
    min_listing_days: int = 120
    min_avg_turnover_20d: float = 100_000_000
    exclude_suspended: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    medium_term_weight: float = 0.6
    timing_weight: float = 0.3
    risk_weight: float = 0.1
    focused_count: int = 5
    candidate_count: int = 20
    lookback_days: int = 120


@dataclass(frozen=True)
class RiskConfig:
    single_position_cap: float = 0.15
    max_focused_holdings: int = 6
    max_daily_new_buys: int = 3
    hard_stop_loss_min: float = -0.10
    hard_stop_loss_max: float = -0.07
    trailing_profit_start_min: float = 0.12
    trailing_profit_start_max: float = 0.20
    portfolio_warning_drawdown: float = -0.12
    portfolio_reduce_risk_drawdown: float = -0.15
    allow_auto_order: bool = False
    allow_margin: bool = False


@dataclass(frozen=True)
class ReportConfig:
    output_markdown: bool = True
    output_excel: bool = True
    output_dir: str = "outputs"


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    rebalance_frequency: str = "daily"


@dataclass(frozen=True)
class ProviderCacheConfig:
    enabled: bool = False
    base_dir: str = "data/raw/akshare"


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "demo"
    stale_days: int = 5
    adjustment: str = "qfq"
    cache: ProviderCacheConfig = field(default_factory=ProviderCacheConfig)

    def __post_init__(self) -> None:
        if isinstance(self.cache, dict):
            object.__setattr__(self, "cache", ProviderCacheConfig(**self.cache))
        if self.adjustment != "qfq":
            raise ValueError(
                "provider.adjustment must be 'qfq' for v0.3 real-data ingestion"
            )


@dataclass(frozen=True)
class DataQualityStatus:
    provider: str
    data_mode: str
    latest_data_date: date | None
    report_date: date
    is_stale: bool
    is_future_dated: bool
    quality_status: str
    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    adjustment: str = "qfq"


@dataclass(frozen=True)
class AppConfig:
    universe: UniverseConfig
    strategy: StrategyConfig
    risk: RiskConfig
    reports: ReportConfig
    backtest: BacktestConfig
    provider: ProviderConfig = field(default_factory=ProviderConfig)


@dataclass(frozen=True)
class StockInfo:
    code: str
    name: str
    board: str
    listing_date: date
    is_st: bool
    is_delisting_risk: bool
    is_suspended: bool = False


@dataclass(frozen=True)
class RiskState:
    current_drawdown: float
    current_holdings: int
    daily_new_buys: int
