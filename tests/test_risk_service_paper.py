import pytest

from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def test_on_fill_skips_account_updates_for_paper():
    account = Account(float("inf"), cash=0.0)
    account.update_position("BTC", 1.0, price=100.0)

    calls = {"pos": 0, "open": 0}

    account.update_position_orig = account.update_position
    account.update_open_order_orig = account.update_open_order

    def mock_update_position(symbol, delta, price=None):
        calls["pos"] += 1
        account.update_position_orig(symbol, delta, price)

    def mock_update_open_order(symbol, side, delta_qty):
        calls["open"] += 1
        account.update_open_order_orig(symbol, side, delta_qty)

    account.update_position = mock_update_position
    account.update_open_order = mock_update_open_order

    guard = PortfolioGuard(GuardConfig(venue="test"))
    rs = RiskService(guard, account=account)

    rs.on_fill("BTC", "buy", 1.0, price=100.0, venue="paper")

    assert calls["pos"] == 0
    assert calls["open"] == 0
    assert account.positions["BTC"] == pytest.approx(1.0)


def test_complete_order_keeps_other_open_orders():
    guard = PortfolioGuard(GuardConfig(venue="paper"))
    account = Account(float("inf"), cash=0.0)
    account.mark_price("BTC", 100.0)
    svc = RiskService(guard, account=account)

    account.update_open_order("BTC", "buy", 1.0)
    account.update_open_order("BTC", "sell", 2.0)

    locked_before = sum(
        svc.account.get_locked_usd(sym) for sym in svc.account.open_orders
    )
    assert locked_before == pytest.approx(300.0)

    svc.complete_order(venue="paper", symbol="BTC", side="buy")

    remaining = svc.account.open_orders.get("BTC", {})
    assert remaining.get("sell") == pytest.approx(2.0)
    assert "buy" not in remaining

    locked_after = sum(
        svc.account.get_locked_usd(sym) for sym in svc.account.open_orders
    )
    assert locked_after == pytest.approx(200.0)
