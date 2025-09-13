import pytest

from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


def test_allow_short_false_caps_to_inventory():
    """Selling more than the current position should not create a short.

    When ``allow_short`` is disabled the risk service should allow liquidating
    at most the owned quantity and clamp the delta accordingly.
    """
    account = Account(float("inf"), cash=1000.0)
    guard = PortfolioGuard(GuardConfig(venue="test"))
    rs = RiskService(guard, account=account, risk_pct=0.0, risk_per_trade=1.0)
    rs.allow_short = False

    account.update_position("ETH", 0.5)

    allowed, _, delta = rs.check_order("ETH", "sell", 100.0)
    assert allowed is True
    assert delta == pytest.approx(-0.5)
