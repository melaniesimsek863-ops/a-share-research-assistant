from pathlib import Path

import pytest

from a_share_ai.config import load_config
from a_share_ai.models import ProviderConfig


def test_load_default_config_has_required_risk_limits():
    config = load_config(Path("configs/default.yaml"))

    assert config.universe.include_boards == ["main", "chinext"]
    assert config.universe.exclude_star_market is True
    assert config.universe.min_listing_days == 120
    assert config.risk.single_position_cap == 0.15
    assert config.risk.max_focused_holdings == 6
    assert config.risk.portfolio_warning_drawdown == -0.12
    assert config.risk.portfolio_reduce_risk_drawdown == -0.15
    assert config.risk.allow_auto_order is False
    assert config.reports.output_markdown is True
    assert config.reports.output_excel is True


def test_provider_config_defaults() -> None:
    config = ProviderConfig()

    assert config.name == "demo"
    assert config.stale_days == 5
    assert config.adjustment == "qfq"


def test_load_config_reads_provider_section(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
provider:
  name: akshare
  stale_days: 3
  adjustment: qfq
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.provider.name == "akshare"
    assert config.provider.stale_days == 3
    assert config.provider.adjustment == "qfq"


def test_load_config_reads_provider_cache_section(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
provider:
  name: akshare
  adjustment: qfq
  stale_days: 5
  cache:
    enabled: true
    base_dir: custom/cache
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.provider.cache.enabled is True
    assert config.provider.cache.base_dir == "custom/cache"


def test_load_config_rejects_non_qfq_adjustment(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
provider:
  name: akshare
  adjustment: hfq
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="qfq"):
        load_config(path)
