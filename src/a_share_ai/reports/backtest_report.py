from __future__ import annotations

from a_share_ai.models import (
    DATA_MODE_REAL_PUBLIC,
    DATA_MODE_SYNTHETIC_DEMO,
    DataQualityStatus,
)


def build_backtest_summary(
    metrics: dict[str, float | int],
    *,
    data_mode: str | None = None,
    latest_data_date: object | None = None,
    data_warning: str | None = None,
    data_quality: DataQualityStatus | None = None,
) -> str:
    lines = ["# 基础回测摘要（烟雾测试）", ""]

    lines.append(
        "> \u8be5\u6458\u8981\u662f\u65e5\u62a5\u94fe\u8def\u70df\u96fe\u6d4b\u8bd5\uff0c\u4e0d\u662f\u5386\u53f2\u591a\u65e5\u671f\u7b56\u7565\u6709\u6548\u6027\u8bc1\u660e\u3002"
    )
    if data_quality is not None:
        banner = (
            "REAL PUBLIC MARKET DATA / \u516c\u5f00\u5e02\u573a\u6570\u636e"
            if data_quality.data_mode == DATA_MODE_REAL_PUBLIC
            else "SYNTHETIC DEMO DATA / \u5408\u6210\u6f14\u793a\u6570\u636e"
            if data_quality.data_mode == DATA_MODE_SYNTHETIC_DEMO
            else "DATA QUALITY METADATA / \u6570\u636e\u8d28\u91cf\u8bf4\u660e"
        )
        lines.extend(
            [
                f"> **{banner}**",
                f"> provider: {data_quality.provider}; data_mode: {data_quality.data_mode}; ",
                f"latest_data_date: {data_quality.latest_data_date}; report_date: {data_quality.report_date}; ",
                f"adjustment: {data_quality.adjustment}; quality_status: {data_quality.quality_status}",
                f"> warnings: {', '.join(data_quality.warnings) or 'none'}; ",
                f"blocking_reasons: {', '.join(data_quality.blocking_reasons) or 'none'}",
                "",
            ]
        )
    if data_quality is None and data_mode == "synthetic_demo":
        lines.extend(["> **SYNTHETIC DEMO DATA / 合成演示数据**", ""])
    if latest_data_date is not None:
        lines.extend([f"- 最新数据日期：{latest_data_date}", ""])

    lines.extend(
        [
            "> 本摘要只检查同日报告链路、交易成本和指标输出是否可运行；不代表未来收益。",
            "> **它不是历史多日期策略回测证据，不可用于证明策略有效性。**",
        ]
    )
    if data_warning:
        lines.append(f"> 数据警告：{data_warning}")

    lines.extend(
        [
            "",
            f"- 总收益率：{metrics['total_return']:.2%}",
            f"- 最大回撤：{metrics['max_drawdown']:.2%}",
            f"- 胜率：{metrics['win_rate']:.2%}",
            f"- 盈亏比：{metrics['payoff_ratio']}",
            f"- 交易次数：{metrics['trade_count']}",
            f"- 平均持仓天数：{metrics['avg_holding_days']}",
            "",
        ]
    )
    return "\n".join(lines)
