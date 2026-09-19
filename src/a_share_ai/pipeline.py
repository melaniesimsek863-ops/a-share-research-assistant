from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from a_share_ai.ai_agent.analyst import explain_candidate
from a_share_ai.backtest.engine import run_simple_backtest
from a_share_ai.config import load_config
from a_share_ai.data.providers import MarketDataProvider, create_provider
from a_share_ai.data.quality import (
    CACHE_REFRESH_FAILED_COLUMN,
    apply_quality_block,
    assess_data_quality,
    latest_data_date as find_latest_data_date,
    remove_future_bars,
    validate_bar_data,
)
from a_share_ai.data.universe import filter_universe
from a_share_ai.features.technical import add_technical_features
from a_share_ai.models import QUALITY_BLOCKED, RiskState
from a_share_ai.reports.backtest_report import build_backtest_summary
from a_share_ai.reports.excel_report import write_excel_signals
from a_share_ai.reports.markdown_report import build_markdown_report
from a_share_ai.risk.rules import apply_risk_rules
from a_share_ai.strategies.medium_term import score_medium_term
from a_share_ai.strategies.ranking import rank_candidates
from a_share_ai.strategies.timing import score_timing



def _force_observation_only(candidates: pd.DataFrame) -> pd.DataFrame:
    guarded = candidates.copy()
    if guarded.empty:
        return guarded

    note = "Forced observation-only safety guard."
    guarded["can_buy"] = False
    guarded["suggested_position"] = 0.0
    guarded["risk_action"] = "forced_observation_only"
    existing_notes = guarded.get(
        "risk_notes", pd.Series([""] * len(guarded), index=guarded.index)
    ).fillna("")
    guarded["risk_notes"] = existing_notes.map(
        lambda value: f"{value}; {note}".strip("; ")
    )
    return guarded

def run_daily_pipeline(
    config_path: str | Path,
    report_date: date,
    provider: MarketDataProvider | None = None,
    provider_name: str | None = None,
    output_dir: str | Path | None = None,
    risk_state: RiskState | None = None,
    force_observation_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_provider_name = provider_name or config.provider.name
    provider = provider or create_provider(
        selected_provider_name,
        adjustment=config.provider.adjustment,
        cache_enabled=config.provider.cache.enabled,
        cache_base_dir=config.provider.cache.base_dir,
        report_date=report_date,
    )
    output_base = Path(output_dir or config.reports.output_dir)
    data_mode = getattr(provider, "data_mode", "unspecified")
    uses_default_risk_state = risk_state is None
    effective_risk_state = risk_state or RiskState(
        current_drawdown=0.0,
        current_holdings=0,
        daily_new_buys=0,
    )

    stock_info = provider.get_stock_info()
    preliminary_eligible, preliminary_excluded = filter_universe(
        stock_info,
        report_date,
        config.universe,
        check_liquidity=False,
    )
    raw_price_history = provider.get_price_history(
        preliminary_eligible["code"].tolist()
    )
    source_cache_refresh_failed = (
        CACHE_REFRESH_FAILED_COLUMN in raw_price_history.columns
        and raw_price_history[CACHE_REFRESH_FAILED_COLUMN].fillna(False).astype(bool).any()
    )
    _, source_had_future = remove_future_bars(raw_price_history, report_date)
    validated_history, bar_warnings, bar_blocking_reasons = validate_bar_data(
        raw_price_history
    )
    source_latest_data_date = find_latest_data_date(validated_history)
    quality_price_history = validated_history
    price_history, _ = remove_future_bars(validated_history, report_date)

    stock_with_liquidity = preliminary_eligible.copy()
    if price_history.empty:
        derived_turnover = pd.Series(dtype=float)
    else:
        last_twenty = (
            price_history.sort_values(["code", "date"])
            .groupby("code", group_keys=False)
            .tail(20)
        )
        derived_turnover = last_twenty.groupby("code")["amount"].mean()
    existing_turnover = pd.to_numeric(
        stock_with_liquidity["avg_turnover_20d"], errors="coerce"
    )
    stock_with_liquidity["avg_turnover_20d"] = existing_turnover.where(
        existing_turnover.notna(),
        stock_with_liquidity["code"].map(derived_turnover),
    )
    eligible, liquidity_excluded = filter_universe(
        stock_with_liquidity,
        report_date,
        config.universe,
    )
    excluded = pd.concat(
        [preliminary_excluded, liquidity_excluded], ignore_index=True
    )
    price_history = price_history[
        price_history["code"].isin(eligible["code"])
    ].reset_index(drop=True)

    stale_demo_data = (
        data_mode == "synthetic_demo"
        and source_latest_data_date is not None
        and report_date > source_latest_data_date
    )
    blocking_reasons = list(bar_blocking_reasons)
    if source_cache_refresh_failed:
        blocking_reasons.append(CACHE_REFRESH_FAILED_COLUMN)
    if stale_demo_data:
        blocking_reasons.append("stale_demo_data")
    data_quality = assess_data_quality(
        provider=getattr(provider, "provider_name", selected_provider_name),
        data_mode=data_mode,
        price_history=quality_price_history,
        report_date=report_date,
        stale_days=config.provider.stale_days,
        adjustment=config.provider.adjustment,
        warnings=bar_warnings,
        blocking_reasons=blocking_reasons,
        source_had_future=source_had_future,
    )
    latest_data_date = data_quality.latest_data_date
    enriched = add_technical_features(price_history)
    latest = enriched.sort_values("date").groupby("code", as_index=False).tail(1)
    latest = latest.merge(eligible[["code", "name"]], on="code", how="left")

    scored = score_medium_term(latest)
    scored = score_timing(scored)
    ranked = rank_candidates(scored, config.strategy)
    candidates = apply_risk_rules(
        ranked,
        config.risk,
        effective_risk_state,
    )
    if data_quality.quality_status == QUALITY_BLOCKED:
        candidates = apply_quality_block(candidates, data_quality)
    if stale_demo_data:
        data_warning = (
            f"报告日期晚于合成演示数据最新日期 {source_latest_data_date.isoformat()}；"
            "不生成可买建议。"
        )
        candidates["risk_action"] = "stale_demo_data"
        candidates["risk_notes"] = data_warning
    elif data_mode == "synthetic_demo":
        data_warning = "固定合成行情仅用于离线流程演示，不是实时或历史实盘数据。"
    else:
        data_warning = None
    if force_observation_only:
        candidates = _force_observation_only(candidates)
        data_warning = "Forced observation-only safety guard; no buy recommendations."


    if candidates.empty:
        candidates["analysis"] = pd.Series(index=candidates.index, dtype=object)
    else:
        candidates["analysis"] = candidates.apply(lambda row: explain_candidate(row), axis=1)

    markdown = build_markdown_report(
        candidates,
        excluded,
        report_date,
        data_mode=data_mode,
        latest_data_date=latest_data_date,
        data_warning=data_warning,
        data_quality=data_quality,
        uses_default_risk_state=uses_default_risk_state,
        risk_state=effective_risk_state,
    )
    markdown_path = output_base / "reports" / f"{report_date.isoformat()}_daily_report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    excel_path = output_base / "signals" / f"{report_date.isoformat()}_signals.xlsx"
    output_metadata = {
        "data_quality": data_quality,
        "data_mode": data_mode,
        "latest_data_date": latest_data_date,
        "data_warning": data_warning or "",
        "uses_default_risk_state": uses_default_risk_state,
        "risk_state_source": "default_demo" if uses_default_risk_state else "caller_supplied",
        "risk_state_summary": (
            f"current_drawdown={effective_risk_state.current_drawdown}, "
            f"current_holdings={effective_risk_state.current_holdings}, "
            f"daily_new_buys={effective_risk_state.daily_new_buys}"
        ),
    }
    write_excel_signals(candidates, excel_path, metadata=output_metadata)

    signal_rows = candidates.assign(date=report_date)
    _, _, backtest_metrics = run_simple_backtest(signal_rows, price_history, config.backtest)
    backtest_text = build_backtest_summary(
        backtest_metrics,
        data_mode=data_mode,
        latest_data_date=latest_data_date,
        data_warning=data_warning,
        data_quality=data_quality,
    )
    backtest_path = output_base / "backtest" / "backtest_summary.md"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_path.write_text(backtest_text, encoding="utf-8")

    return {
        "markdown_path": markdown_path,
        "excel_path": excel_path,
        "backtest_path": backtest_path,
        "candidates": candidates,
        "excluded": excluded,
        "data_mode": data_mode,
        "latest_data_date": latest_data_date,
        "data_warning": data_warning,
        "data_quality": data_quality,
        "provider": data_quality.provider,
        "risk_state": effective_risk_state,
    }
