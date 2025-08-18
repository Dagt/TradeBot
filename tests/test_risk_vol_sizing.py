import pytest
import numpy as np

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService
from tradingbot.risk.position_sizing import vol_target


def test_vol_target_function(synthetic_volatility):
    size = vol_target(synthetic_volatility, risk_budget=0.2, notional_cap=10)
    assert size == pytest.approx(5.0)


def test_vol_target_respects_cap(synthetic_volatility):
    size = vol_target(synthetic_volatility, risk_budget=1.0, notional_cap=10)
    assert size == pytest.approx(10.0)


def test_size_with_volatility_calls_vol_target(monkeypatch, synthetic_volatility):
    called = {}

    def fake_vol_target(atr, risk_budget, notional_cap):
        called["args"] = (atr, risk_budget, notional_cap)
        return 3.0

    monkeypatch.setattr("tradingbot.risk.manager._vol_target", fake_vol_target)

    rm = RiskManager(max_pos=10, vol_target=0.02)
    delta = rm.size_with_volatility(synthetic_volatility)

    assert called["args"] == (
        synthetic_volatility,
        rm.max_pos * rm.vol_target,
        rm.max_pos,
    )
    assert delta == pytest.approx(3.0)


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
