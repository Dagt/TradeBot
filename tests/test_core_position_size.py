import pytest

from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


def test_core_calc_position_size_scales_with_strength():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account, risk_per_trade=0.1)
    price = 100.0
    full = rm.calc_position_size(1.0, price)
    partial = rm.calc_position_size(0.37, price)
    assert partial == pytest.approx(full * 0.37)


def test_service_calc_position_size_passes_strength():
    account = Account(float("inf"), cash=1000.0)
    rm = RiskManager()
    guard = PortfolioGuard(GuardConfig(venue="test"))
    svc = RiskService(rm, guard, account=account, risk_per_trade=0.1)
    price = 100.0
    full = svc.calc_position_size(1.0, price)
    partial = svc.calc_position_size(0.37, price)
    assert partial == pytest.approx(full * 0.37)
