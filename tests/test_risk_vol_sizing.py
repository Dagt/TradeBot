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
    rm = RiskManager(equity_pct=0.1, vol_target=0.02)
    price = 1.0
    delta = rm.size("buy", price, equity, symbol="BTC", symbol_vol=synthetic_volatility)
    budget = equity * rm.equity_pct
    expected = budget / price + budget / synthetic_volatility
    assert delta == pytest.approx(expected)


def test_vol_target_basic():
    assert vol_target(atr=2.0, equity_pct=1.0, equity=10.0) == pytest.approx(5.0)


def test_vol_target_scales_linearly():
    assert vol_target(atr=1.0, equity_pct=0.5, equity=20.0) == pytest.approx(10.0)


def test_risk_vol_sizing_with_correlation(synthetic_volatility):
    equity = 1.0
    rm = RiskManager(equity_pct=0.1, vol_target=0.02)
    corr = {("BTC", "ETH"): 0.9}
    price = 1.0
    delta = rm.size(
        "buy",
        price,
        equity,
        symbol="BTC",
        symbol_vol=synthetic_volatility,
        correlations=corr,
        threshold=0.8,
    )
    budget = equity * rm.equity_pct
    expected = (budget / price + budget / synthetic_volatility) * 0.5
    assert delta == pytest.approx(expected)


def test_risk_service_uses_guard_volatility():
    guard = PortfolioGuard(GuardConfig(per_symbol_cap_pct=10000, total_cap_pct=20000))
    rm = RiskManager(equity_pct=0.1, vol_target=0.02)
    rm_guard_equity = 1.0
    guard.refresh_usd_caps(rm_guard_equity)
    svc = RiskService(rm, guard)
    guard.st.returns["BTC"].extend([0.01, -0.02, 0.03])
    allowed, _, delta = svc.check_order("BTC", "buy", 1.0, 1.0)
    vol = np.std([0.01, -0.02, 0.03]) * np.sqrt(365)
    budget = rm_guard_equity * rm.equity_pct
    expected = budget / 1.0 + budget / vol
    assert allowed
    assert delta == pytest.approx(expected)
