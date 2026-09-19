from __future__ import annotations

from datetime import date

import pandas as pd

from a_share_ai.models import (
    DATA_MODE_REAL_PUBLIC,
    DATA_MODE_SYNTHETIC_DEMO,
    DataQualityStatus,
)


def _labels_text(value: object) -> str:
    if isinstance(value, list):
        return "、".join(value)
    return str(value)


def build_markdown_report(
    candidates: pd.DataFrame,
    excluded: pd.DataFrame,
    report_date: date,
    *,
    data_mode: str | None = None,
    latest_data_date: date | None = None,
    data_warning: str | None = None,
    data_quality: DataQualityStatus | None = None,
    uses_default_risk_state: bool = False,
    risk_state: object | None = None,
) -> str:
    lines = [
        f"# A股短中线交易辅助日报 - {report_date.isoformat()}",
        "> EXIT RULE STATUS: RESERVED / NOT ACTIVE (stop-loss and trailing-profit are not executed by candidate generation).",
        "",
        "",
        "> 本报告是研究辅助，不是收益承诺；所有交易必须人工确认。",
        "",
    ]
    if data_quality is not None:
        banner = (
            "REAL PUBLIC MARKET DATA / \u516c\u5f00\u5e02\u573a\u6570\u636e"
            if data_quality.data_mode == DATA_MODE_REAL_PUBLIC
            else "SYNTHETIC DEMO DATA / \u5408\u6210\u6f14\u793a\u6570\u636e"
            if data_quality.data_mode == DATA_MODE_SYNTHETIC_DEMO
            else "DATA QUALITY METADATA / \u6570\u636e\u8d28\u91cf\u8bf4\u660e"
        )
        warnings = ", ".join(data_quality.warnings) or "none"
        blocking_reasons = ", ".join(data_quality.blocking_reasons) or "none"
        lines.extend(
            [
                f"> **{banner}**",
                f"- provider: {data_quality.provider}",
                f"- data_mode: {data_quality.data_mode}",
                f"- latest_data_date: {data_quality.latest_data_date}",
                f"- report_date: {data_quality.report_date}",
                f"- \u590d\u6743\u65b9\u5f0f\uff1a{data_quality.adjustment}",
                f"- quality_status: {data_quality.quality_status}",
                f"- warnings: {warnings}",
                f"- blocking_reasons: {blocking_reasons}",
                "",
            ]
        )
    if data_quality is None and data_mode == "synthetic_demo":
        lines.extend(["> **SYNTHETIC DEMO DATA / 合成演示数据**", ""])
    if latest_data_date is not None:
        lines.extend([f"- 最新数据日期：{latest_data_date.isoformat()}", ""])
    if data_warning:
        lines.extend([f"> 数据警告：{data_warning}", ""])
    if uses_default_risk_state:
        lines.extend(
            ["> 风险状态：未传入真实组合状态，使用默认演示风险状态。", ""]
        )
    if risk_state is not None:
        risk_state_source = "default_demo" if uses_default_risk_state else "caller-supplied"
        lines.extend(
            [
                f"> risk_state_source: {risk_state_source}; "
                f"current_drawdown={risk_state.current_drawdown}; "
                f"current_holdings={risk_state.current_holdings}; "
                f"daily_new_buys={risk_state.daily_new_buys}",
                "",
            ]
        )
    if not candidates.empty and "risk_action" in candidates.columns:
        lines.extend(
            [f"> risk_actions: {', '.join(candidates['risk_action'].astype(str).unique())}", ""]
        )
    lines.extend(
        [
            "> 持仓管理边界：止损/移动止盈参数仅为未来持仓管理模块预留，当前候选生成流程未执行这些退出规则。",
            "",
            "## 重点观察",
            "",
        ]
    )
    focused = candidates[candidates["tier"] == "focused"]
    if focused.empty:
        lines.append("今日没有重点观察标的。")
    for _, row in focused.iterrows():
        position = int(round(float(row["suggested_position"]) * 100))
        lines.extend(
            [
                f"### {row['code']} {row['name']}",
                "",
                f"- 综合评分：{row['total_score']}",
                f"- 信号：{_labels_text(row.get('signal_labels', []))}",
                f"- 建议仓位上限：{position}%",
                f"- 风险提示：{row.get('risk_notes', '')}",
                f"- 可买建议：{'是（仍需人工确认）' if bool(row.get('can_buy', False)) else '否（仅观察）'}",
                "",
            ]
        )

    lines.extend(["## 普通候选", ""])
    normal = candidates[candidates["tier"] == "candidate"]
    if normal.empty:
        lines.append("今日没有普通候选。")
    for _, row in normal.iterrows():
        lines.append(f"- {row['code']} {row['name']}：综合评分 {row['total_score']}")

    lines.extend(["", "## 剔除样例", ""])
    if excluded.empty:
        lines.append("没有剔除样例。")
    for _, row in excluded.head(20).iterrows():
        lines.append(f"- {row['code']} {row['name']}：{row['exclude_reason']}")

    return "\n".join(lines) + "\n"
