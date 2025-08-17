import pytest

from tradingbot.risk.manager import RiskManager


def test_risk_vol_sizing(synthetic_volatility):
    rm = RiskManager(max_pos=10, vol_target=0.02)
    delta = rm.size("buy", symbol="BTC", symbol_vol=synthetic_volatility)
    expected = rm.max_pos + min(
        rm.max_pos, rm.max_pos * rm.vol_target / synthetic_volatility
    )
    assert delta == pytest.approx(expected)


def test_risk_vol_sizing_with_correlation(synthetic_volatility):
    rm = RiskManager(max_pos=10, vol_target=0.02)
    corr = {("BTC", "ETH"): 0.9}
    delta = rm.size(
        "buy",
        symbol="BTC",
        symbol_vol=synthetic_volatility,
        correlations=corr,
        threshold=0.8,
    )
    expected = rm.max_pos + min(
        rm.max_pos, rm.max_pos * rm.vol_target / synthetic_volatility
    )
    expected *= 0.5
    assert delta == pytest.approx(expected)
