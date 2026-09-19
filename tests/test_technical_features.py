from a_share_ai.features.market import calculate_market_regime
from a_share_ai.features.technical import add_technical_features
from tests.fixtures.sample_data import sample_price_history


def test_add_technical_features_creates_expected_columns():
    bars = sample_price_history(days=140, trend=0.002)
    enriched = add_technical_features(bars)

    required = {
        "ma5",
        "ma20",
        "ma60",
        "return_20d",
        "return_60d",
        "volume_ratio_5_20",
        "drawdown_60d",
        "above_ma20",
        "ma_bullish",
    }
    assert required.issubset(enriched.columns)
    latest = enriched.iloc[-1]
    assert latest["ma5"] > latest["ma20"] > latest["ma60"]
    assert latest["above_ma20"] is True
    assert latest["ma_bullish"] is True


def test_market_regime_labels_uptrend():
    index_bars = sample_price_history(code="000300", days=140, trend=0.001)
    regime = calculate_market_regime(index_bars)

    assert regime["label"] == "uptrend"
    assert regime["score"] > 0
