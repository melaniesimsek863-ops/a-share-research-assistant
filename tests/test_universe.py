from datetime import date

from a_share_ai.config import load_config
from a_share_ai.data.universe import filter_universe
from tests.fixtures.sample_data import sample_stock_info


def test_filter_universe_excludes_st_star_new_suspended_and_low_turnover():
    config = load_config("configs/default.yaml")
    eligible, excluded = filter_universe(
        sample_stock_info(),
        as_of=date(2026, 7, 29),
        config=config.universe,
    )

    assert eligible["code"].tolist() == ["000001", "300001"]

    reasons = dict(zip(excluded["code"], excluded["exclude_reason"], strict=True))
    assert reasons["600001"] == "st_or_delisting_risk"
    assert reasons["688001"] == "star_market_excluded"
    assert reasons["000002"] == "listed_less_than_120_days"
    assert reasons["300002"] == "suspended"
    assert reasons["000003"] == "low_liquidity"
