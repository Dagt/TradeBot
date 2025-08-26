import pytest
import numpy as np

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def test_risk_vol_sizing(synthetic_volatility):
    rm = RiskManager(max_pos=10, vol_target=0.02)
    delta = rm.size("buy", symbol="BTC", symbol_vol=synthetic_volatility)
    expected = rm.max_pos + rm.max_pos * rm.vol_target / synthetic_volatility
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
    expected = rm.max_pos + rm.max_pos * rm.vol_target / synthetic_volatility
    expected *= 0.5
    assert delta == pytest.approx(expected)


def test_risk_service_uses_guard_volatility():
    guard = PortfolioGuard(GuardConfig(per_symbol_cap_usdt=10000, total_cap_usdt=20000))
    rm = RiskManager(max_pos=10, vol_target=0.02)
    svc = RiskService(rm, guard)
    guard.st.returns["BTC"].extend([0.01, -0.02, 0.03])
    allowed, _, delta = svc.check_order("BTC", "buy", price=100.0)
    vol = np.std([0.01, -0.02, 0.03]) * np.sqrt(365)
    expected = rm.max_pos + rm.max_pos * rm.vol_target / vol
    assert allowed
    assert delta == pytest.approx(expected)
