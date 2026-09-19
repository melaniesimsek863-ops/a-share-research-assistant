from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def build_replay_summary(
    summary: pd.DataFrame,
    candidate_count: int,
    replay_dates: Iterable[str],
) -> str:
    date_list = [str(date) for date in replay_dates]
    if date_list:
        date_range = f"{date_list[0]} 至 {date_list[-1]}"
    else:
        date_range = "无"

    lines = [
        "# 历史回放验证摘要",
        "",
        "HISTORICAL RESEARCH ONLY / 历史研究用途",
        "NOT A RECOMMENDATION / 不构成投资建议",
        "",
        "本报告只用于研究候选信号在历史样本中的后续表现，回放结果不代表未来收益，也不能直接作为买卖依据。",
        "",
        f"- 回放日期范围：{date_range}",
        f"- \u5b9e\u9645\u56de\u653e\u4ea4\u6613\u65e5\u6570\uff1a{len(date_list)}",
        f"- \u5019\u9009\u8bb0\u5f55\u6570\uff1a{candidate_count}",
        "",
        "| Horizon | 样本数 | 平均收益 | 中位收益 | 胜率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in summary.iterrows():
        lines.append(
            "| {horizon} | {sample_count} | {avg_return} | {median_return} | {win_rate} |".format(
                horizon=int(row["horizon"]),
                sample_count=int(row["sample_count"]),
                avg_return=_format_percent(row["avg_return"]),
                median_return=_format_percent(row["median_return"]),
                win_rate=_format_percent(row["win_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "解读提醒：样本数越少，结果越不稳定；胜率、平均收益和中位收益必须结合数据质量、交易成本、停牌/涨跌停约束和风险敞口一起看。",
        ]
    )
    return "\n".join(lines)


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"
