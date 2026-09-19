from datetime import date

import pandas as pd

from a_share_ai.data.quality import (
    apply_quality_block,
    assess_data_quality,
    remove_future_bars,
    validate_bar_data,
)
from a_share_ai.models import QUALITY_BLOCKED


def test_remove_future_bars_drops_rows_after_report_date() -> None:
    frame = pd.DataFrame(
        [
            {"date": date(2026, 7, 28), "code": "000001", "close": 10.0},
            {"date": date(2026, 7, 30), "code": "000001", "close": 11.0},
        ]
    )

    filtered, had_future = remove_future_bars(frame, date(2026, 7, 29))

    assert had_future is True
    assert filtered["date"].tolist() == [date(2026, 7, 28)]


def test_assess_data_quality_blocks_stale_data() -> None:
    frame = pd.DataFrame([{"date": date(2026, 7, 20), "code": "000001", "close": 10.0}])

    status = assess_data_quality(
        provider="akshare",
        data_mode="real_public_market_data",
        price_history=frame,
        report_date=date(2026, 7, 29),
        stale_days=5,
        adjustment="qfq",
    )

    assert status.latest_data_date == date(2026, 7, 20)
    assert status.is_stale is True
    assert status.quality_status == QUALITY_BLOCKED
    assert "stale_data" in status.blocking_reasons


def test_apply_quality_block_disables_buy_suggestions() -> None:
    candidates = pd.DataFrame(
        [{"code": "000001", "can_buy": True, "risk_action": "allow", "suggested_position": 0.1}]
    )
    status = assess_data_quality(
        provider="akshare",
        data_mode="real_public_market_data",
        price_history=pd.DataFrame([{"date": date(2026, 7, 20), "code": "000001", "close": 10.0}]),
        report_date=date(2026, 7, 29),
        stale_days=5,
        adjustment="qfq",
    )

    result = apply_quality_block(candidates, status)

    assert result.loc[0, "can_buy"] is False
    assert result.loc[0, "suggested_position"] == 0.0
    assert result.loc[0, "risk_action"] == "data_quality_blocked"


def test_assess_data_quality_blocks_future_source_rows() -> None:
    frame = pd.DataFrame(
        [
            {"date": date(2026, 7, 29), "code": "000001", "close": 10.0},
            {"date": date(2026, 7, 30), "code": "000001", "close": 11.0},
        ]
    )

    status = assess_data_quality(
        provider="akshare",
        data_mode="real_public_market_data",
        price_history=frame,
        report_date=date(2026, 7, 29),
        stale_days=5,
        adjustment="qfq",
    )

    assert status.quality_status == QUALITY_BLOCKED
    assert status.is_future_dated is True
    assert "future_data_detected" in status.blocking_reasons


def test_validate_bar_data_blocks_missing_critical_columns() -> None:
    malformed = pd.DataFrame(
        [{"date": date(2026, 7, 29), "code": "000001", "close": 10.0}]
    )

    validated, warnings, blocking_reasons = validate_bar_data(malformed)

    assert validated.empty
    assert warnings == []
    assert blocking_reasons == [
        "missing_bar_columns:amount,high,low,open,volume"
    ]
