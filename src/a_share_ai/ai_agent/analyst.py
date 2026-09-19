from __future__ import annotations

import pandas as pd


def explain_candidate(row: pd.Series) -> dict[str, str]:
    labels = row.get("signal_labels", [])
    if not isinstance(labels, list):
        labels = []
    label_text = "、".join(labels) if labels else "暂无明确短线触发标签"
    position_pct = int(round(float(row.get("suggested_position", 0.0)) * 100))

    return {
        "summary": f"{row['code']} {row['name']}：中线评分{row['medium_term_score']}，短线评分{row['timing_score']}。",
        "medium_term_logic": "中线逻辑来自均线结构、阶段涨幅和60日回撤控制的规则评分。",
        "timing_logic": f"短线触发条件：{label_text}。",
        "risk": str(row.get("risk_notes", "风险状态未提供")),
        "plan": f"建议单票仓位不超过{position_pct}%，所有交易必须人工确认。",
    }