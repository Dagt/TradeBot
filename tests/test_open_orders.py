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


def test_update_open_order_clears_when_offset():
    account = Account(float("inf"), cash=0.0)
    account.update_open_order("BTC", "buy", 1.0)
    account.update_open_order("BTC", "buy", -1.0)
    assert account.open_orders == {}


def test_calc_position_size_accounts_for_open_orders():
    account = Account(float("inf"), cash=1000.0)
    account.mark_price("BTC", 100.0)
    account.update_open_order("BTC", "buy", 2.0)  # reserves $200
    svc = make_service(account)
    size = svc.calc_position_size(1.0, 100.0)
    assert size == pytest.approx(0.8)



def test_check_order_pending_qty_reduces_next_size():
    account = Account(float("inf"), cash=1000.0)
    account.mark_price("BTC", 100.0)
    svc = make_service(account)

    allowed, _, delta = svc.check_order("BTC", "buy", 100.0, strength=1.0)
    assert allowed
    svc.account.update_open_order("BTC", "buy", delta)

    allowed_raw, _, delta_raw = svc.check_order("BTC", "buy", 100.0, strength=1.0)
    assert allowed_raw
    assert delta_raw > 0

    allowed2, reason2, delta2 = svc.check_order(
        "BTC",
        "buy",
        100.0,
        strength=1.0,
        pending_qty=svc.account.open_orders.get("BTC", {}).get("buy", 0.0),
    )
    assert not allowed2
    assert reason2 == "below_min_qty"
    assert delta2 == pytest.approx(0.0)


def test_check_order_pending_qty_handles_partial_fill():
    account = Account(float("inf"), cash=1000.0)
    account.mark_price("BTC", 100.0)
    svc = make_service(account)

    allowed, _, delta = svc.check_order("BTC", "buy", 100.0, strength=1.0)
    assert allowed
    svc.account.update_open_order("BTC", "buy", delta)
    # simulate a partial fill of half the order
    svc.on_fill("BTC", "buy", delta / 2, price=100.0)
    pending = svc.account.open_orders.get("BTC", {}).get("buy", 0.0)
    base = svc.calc_position_size(1.0, 100.0)

    allowed2, _, delta2 = svc.check_order(
        "BTC",
        "buy",
        100.0,
        strength=1.0,
        pending_qty=pending,
    )
    assert allowed2
    assert delta2 == pytest.approx(base - pending)


def test_available_balance_decreases_with_pending_order():
    account = Account(float("inf"), cash=1000.0)
    account.mark_price("BTC", 100.0)
    svc = make_service(account)
    allowed, _, delta = svc.check_order("BTC", "buy", 100.0, strength=1.0)
    assert allowed
    account.update_open_order("BTC", "buy", delta)
    assert account.cash == pytest.approx(1000.0)
    assert account.get_available_balance() == pytest.approx(900.0)
