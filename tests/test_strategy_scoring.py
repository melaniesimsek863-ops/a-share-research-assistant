import pandas as pd

from a_share_ai.config import load_config
from a_share_ai.ai_agent.analyst import explain_candidate
from a_share_ai.strategies.medium_term import score_medium_term
from a_share_ai.strategies.ranking import rank_candidates
from a_share_ai.strategies.timing import score_timing


def test_strategy_scores_and_tiers_candidates():
    rows = pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "强趋势",
                "close": 15.0,
                "ma5": 14.8,
                "ma20": 14.0,
                "ma60": 12.0,
                "return_20d": 0.12,
                "return_60d": 0.28,
                "volume_ratio_5_20": 1.6,
                "drawdown_60d": -0.03,
                "above_ma20": True,
                "ma_bullish": True,
            },
            {
                "code": "300001",
                "name": "普通趋势",
                "close": 20.0,
                "ma5": 19.5,
                "ma20": 19.0,
                "ma60": 18.0,
                "return_20d": 0.03,
                "return_60d": 0.09,
                "volume_ratio_5_20": 1.1,
                "drawdown_60d": -0.08,
                "above_ma20": True,
                "ma_bullish": True,
            },
        ]
    )
    config = load_config("configs/default.yaml")

    medium = score_medium_term(rows)
    timed = score_timing(medium)
    ranked = rank_candidates(timed, config.strategy)

    assert ranked.iloc[0]["code"] == "000001"
    assert ranked.iloc[0]["tier"] == "focused"
    assert ranked.iloc[0]["total_score"] > ranked.iloc[1]["total_score"]
    assert "放量且收盘在20日均线上方" in ranked.iloc[0]["signal_labels"]


def test_explain_candidate_uses_structured_fields_only():
    row = pd.Series(
        {
            "code": "000001",
            "name": "强趋势",
            "medium_term_score": 92.0,
            "timing_score": 90.0,
            "signal_labels": ["放量突破", "未明显追高"],
            "risk_notes": "风险状态正常",
            "suggested_position": 0.15,
        }
    )
    explanation = explain_candidate(row)

    assert explanation["summary"] == "000001 强趋势：中线评分92.0，短线评分90.0。"
    assert "放量突破、未明显追高" in explanation["timing_logic"]
    assert "15%" in explanation["plan"]
    assert "新闻" not in explanation["summary"]

def test_explanation_names_only_features_used_by_medium_term_scoring():
    row = pd.Series(
        {
            "code": "000001",
            "name": "演示标的",
            "medium_term_score": 80.0,
            "timing_score": 60.0,
            "signal_labels": [],
            "risk_notes": "风险状态正常",
            "suggested_position": 0.15,
        }
    )

    explanation = explain_candidate(row)

    assert explanation["medium_term_logic"] == "中线逻辑来自均线结构、阶段涨幅和60日回撤控制的规则评分。"
    assert "成交额稳定性" not in explanation["medium_term_logic"]


def test_timing_labels_do_not_claim_unchecked_breakout_or_ma_proximity():
    rows = pd.DataFrame(
        [{"close": 20.0, "ma20": 10.0, "volume_ratio_5_20": 1.6, "drawdown_60d": -0.03, "return_20d": 0.10}]
    )

    labels = score_timing(rows).iloc[0]["signal_labels"]

    assert labels == [
        "放量且收盘在20日均线上方",
        "收盘在20日均线上方且60日回撤大于-8%",
        "20日涨幅低于18%",
    ]
    assert all("突破" not in label and "附近企稳" not in label for label in labels)
