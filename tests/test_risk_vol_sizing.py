import pytest
import numpy as np

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService
from tradingbot.risk.position_sizing import vol_target


def test_risk_vol_sizing(synthetic_volatility):
    rm = RiskManager(max_pos=10, vol_target=0.02)
    delta = rm.size("buy", symbol="BTC", symbol_vol=synthetic_volatility)
    expected = rm.max_pos + min(
        rm.max_pos, rm.max_pos * rm.vol_target / synthetic_volatility
    )
    assert delta == pytest.approx(expected)


def test_vol_target_basic():
    assert vol_target(atr=2.0, risk_budget=10.0, notional_cap=20.0) == pytest.approx(5.0)


def test_vol_target_caps_notional():
    assert vol_target(atr=1.0, risk_budget=50.0, notional_cap=30.0) == pytest.approx(30.0)


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


def test_risk_service_uses_guard_volatility():
    guard = PortfolioGuard(GuardConfig(per_symbol_cap_usdt=10000, total_cap_usdt=20000))
    rm = RiskManager(max_pos=10, vol_target=0.02)
    svc = RiskService(rm, guard)
    guard.st.returns["BTC"].extend([0.01, -0.02, 0.03])
    allowed, _, delta = svc.check_order("BTC", "buy", price=100.0)
    vol = np.std([0.01, -0.02, 0.03]) * np.sqrt(365)
    expected = rm.max_pos + min(rm.max_pos, rm.max_pos * rm.vol_target / vol)
    assert allowed
    assert delta == pytest.approx(expected)
