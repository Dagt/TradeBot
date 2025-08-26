import pytest

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


@pytest.mark.parametrize(
    "cur_qty, strength, expected",
    [
        (0.0, 0.2, 2.0),  # build new position
        (2.0, 0.1, -1.0),  # scale down
    ],
)
def test_equity_based_sizing(cur_qty, strength, expected):
    rm = RiskManager(max_pos=100.0)
    guard = PortfolioGuard(
        GuardConfig(total_cap_usdt=10000.0, per_symbol_cap_usdt=10000.0, venue="test")
    )
    svc = RiskService(rm, guard)
    guard.mark_price("BTC", 100.0)
    svc.update_position("test", "BTC", cur_qty)

    allowed, reason, delta = svc.check_order(
        "BTC", "buy", price=100.0, equity=1000.0, strength=strength
    )
    assert allowed
    assert delta == pytest.approx(expected)


def test_equity_cap_limits_order_size():
    rm = RiskManager(max_pos=100.0)
    guard = PortfolioGuard(
        GuardConfig(total_cap_usdt=10000.0, per_symbol_cap_usdt=10000.0, venue="test")
    )
    svc = RiskService(rm, guard)
    guard.mark_price("BTC", 100.0)
    guard.mark_price("ETH", 100.0)
    # simulate existing exposure of 900 usdt in ETH
    svc.update_position("test", "ETH", 9.0)

    allowed, reason, delta = svc.check_order(
        "BTC", "buy", price=100.0, equity=1000.0, strength=0.5
    )
    assert allowed
    # free equity is 100 -> qty adjusted to 1
    assert delta == pytest.approx(1.0)
