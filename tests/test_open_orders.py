import pytest

from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


def make_service(account: Account) -> RiskService:
    guard = PortfolioGuard(GuardConfig(venue="test"))
    return RiskService(
        guard,
        account=account,
        risk_per_trade=0.1,
        atr_mult=2.0,
        risk_pct=0.01,
    )


def test_calc_position_size_accounts_for_open_orders():
    account = Account(float("inf"), cash=1000.0)
    account.mark_price("BTC", 100.0)
    account.update_open_order("BTC", 2.0)  # reserves $200
    svc = make_service(account)
    size = svc.calc_position_size(1.0, 100.0)
    assert size == pytest.approx(0.8)


def test_check_global_exposure_includes_pending():
    account = Account(max_symbol_exposure=1000.0, cash=0.0)
    account.update_position("BTC", 2.0, price=100.0)  # $200 exposure
    account.mark_price("BTC", 100.0)
    account.update_open_order("BTC", 3.0)  # $300 pending
    svc = make_service(account)
    assert svc.check_global_exposure("BTC", 400.0)
    assert not svc.check_global_exposure("BTC", 600.0)
