import pytest

from tradingbot.core import Account
from tradingbot.risk.position_sizing import delta_from_strength
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def test_delta_from_strength_respects_risk_pct():
    delta = delta_from_strength(
        2.0, equity=1000.0, price=10.0, current_qty=0.0, risk_pct=0.1
    )
    assert delta == pytest.approx(10.0)


def test_risk_manager_caps_position_via_risk_pct():
    account = Account(float("inf"), cash=1000.0)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="test")
    )
    rs = RiskService(guard, account=account, risk_pct=0.1, risk_per_trade=1.0)
    rs.update_position("test", "BTC", 5.0, entry_price=10.0)
    allowed, reason, delta = rs.check_order("BTC", "buy", 10.0, strength=1.0)
    assert allowed
    assert reason == ""
    assert delta == pytest.approx(5.0)
