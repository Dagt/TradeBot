import pytest
import numpy as np

from tradingbot.risk.manager import RiskManager
import numpy as np
import pytest

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService
from tradingbot.risk.position_sizing import vol_target


def test_risk_vol_sizing(synthetic_volatility):
    equity = 1.0
    rm = RiskManager(vol_target=0.02)
    rm.equity_pct = 1.0
    price = 1.0
    base = rm.size("buy", price, equity, strength=0.1)
    delta = rm.size(
        "buy",
        price,
        equity,
        strength=0.1,
        symbol="BTC",
        symbol_vol=synthetic_volatility,
    )
    expected = base + rm.size_with_volatility(synthetic_volatility, price, equity)
    assert delta == pytest.approx(expected)


def test_vol_target_basic():
    assert vol_target(atr=2.0, equity=10.0, vol_target=1.0) == pytest.approx(5.0)


def test_vol_target_scales_linearly():
    assert vol_target(atr=1.0, equity=20.0, vol_target=0.5) == pytest.approx(10.0)


def test_risk_vol_sizing_with_correlation(synthetic_volatility):
    equity = 1.0
    rm = RiskManager(vol_target=0.02)
    rm.equity_pct = 1.0
    corr = {("BTC", "ETH"): 0.9}
    price = 1.0
    base = rm.size(
        "buy",
        price,
        equity,
        strength=0.1,
        symbol="BTC",
        symbol_vol=synthetic_volatility,
    )
    delta = rm.size(
        "buy",
        price,
        equity,
        strength=0.1,
        symbol="BTC",
        symbol_vol=synthetic_volatility,
        correlations=corr,
        threshold=0.8,
    )
    assert delta == pytest.approx(base * 0.5)


def test_risk_service_uses_guard_volatility():
    guard = PortfolioGuard(GuardConfig(per_symbol_cap_pct=10000, total_cap_pct=20000))
    rm = RiskManager(vol_target=0.02)
    rm.equity_pct = 1.0
    rm_guard_equity = 1.0
    guard.refresh_usd_caps(rm_guard_equity)
    svc = RiskService(rm, guard)
    guard.st.returns["BTC"].extend([0.01, -0.02, 0.03])
    base = rm.size(
        "buy", 1.0, 1.0, strength=0.1, symbol="BTC", symbol_vol=guard.volatility("BTC")
    )
    allowed, _, delta = svc.check_order("BTC", "buy", 1.0, 1.0, strength=0.1)
    assert allowed
    assert delta == pytest.approx(base)
