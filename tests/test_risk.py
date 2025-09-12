import asyncio
import pytest

from tradingbot.core import Account
from tradingbot.risk.exceptions import StopLossExceeded
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def _make_rs(equity: float, risk_pct: float = 0.0) -> RiskService:
    account = Account(float("inf"), cash=equity)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="test")
    )
    return RiskService(guard, account=account, risk_pct=risk_pct, risk_per_trade=1.0)


def test_size_scales_with_equity_and_strength():
    price = 100.0
    equity_small = 10_000.0
    equity_big = 20_000.0
    rs_small = _make_rs(equity_small)
    rs_big = _make_rs(equity_big)

    expected_small = equity_small * 0.5 / price
    expected_big = equity_big * 0.5 / price
    assert rs_small.calc_position_size(0.5, price) == pytest.approx(expected_small)
    assert rs_big.calc_position_size(0.5, price) == pytest.approx(expected_big)


def test_stop_loss_risk_pct():
    equity = 10_000.0
    risk_pct = 0.02
    price = 100.0

    qty = equity * 0.10 / price
    rs = _make_rs(equity, risk_pct=risk_pct)
    rs.add_fill("buy", qty, price)

    assert rs.check_limits(price)
    with pytest.raises(StopLossExceeded):
        rs.check_limits(price * (1 - risk_pct - 0.01))
    assert rs.enabled is True
    assert rs.last_kill_reason == "stop_loss"


def test_pyramiding_and_scaling(risk_service):
    rs = risk_service
    price = 100.0
    symbol = "SYM"
    rs.risk_pct = 0.0
    max_qty = rs.account.cash / price

    delta = rs.calc_position_size(0.5, price)
    rs.add_fill("buy", delta, price)
    rs.update_position("test", symbol, rs.pos.qty, entry_price=price)
    assert rs.account.positions[symbol] == pytest.approx(max_qty * 0.5)

    target = rs.calc_position_size(1.0, price)
    rs.add_fill("buy", target - rs.pos.qty, price)
    rs.update_position("test", symbol, rs.pos.qty, entry_price=price)
    assert rs.account.positions[symbol] == pytest.approx(max_qty)

    target = rs.calc_position_size(0.5, price)
    rs.add_fill("sell", rs.pos.qty - target, price)
    rs.update_position("test", symbol, rs.pos.qty, entry_price=price)
    assert rs.account.positions[symbol] == pytest.approx(max_qty * 0.5)

    target = rs.calc_position_size(0.0, price)
    rs.add_fill("sell", rs.pos.qty - target, price)
    rs.update_position("test", symbol, rs.pos.qty, entry_price=price)
    assert rs.account.positions[symbol] == pytest.approx(0.0)


def test_update_position_sets_initial_stop_and_trailing():
    guard = PortfolioGuard(GuardConfig(venue="test"))
    rs = RiskService(guard, account=Account(float("inf")), risk_pct=0.02, risk_per_trade=1.0)
    price = 100.0
    rs.update_position("X", "BTC", 1.0, entry_price=price)
    trade = rs.get_trade("BTC")
    assert trade["stop"] == pytest.approx(98.0)
    assert trade["stage"] == 0

    trade["atr"] = 5.0
    rs.update_trailing(trade, 101.0)
    assert trade["stop"] == pytest.approx(99.0)
    assert trade["stage"] == 1
    rs.update_trailing(trade, 102.0)
    assert trade["stop"] == pytest.approx(100.0)
    assert trade["stage"] == 2
    rs.update_trailing(trade, 103.0)
    assert trade["stop"] == pytest.approx(101.0)
    assert trade["stage"] == 3
    rs.update_trailing(trade, 112.0)
    assert trade["stop"] == pytest.approx(102.0)
    assert trade["stage"] >= 4


@pytest.mark.parametrize("qty", [1.0, -1.0])
def test_update_position_uses_risk_pct_for_stop(qty):
    """Initial stop should be placed at ``risk_pct`` distance from entry."""
    risk_pct = 0.02
    price = 100.0
    expected = price * (1 - risk_pct) if qty > 0 else price * (1 + risk_pct)
    guard = PortfolioGuard(GuardConfig(venue="test"))
    rs = RiskService(
        guard, account=Account(float("inf")), risk_pct=risk_pct, risk_per_trade=1.0
    )
    rs.update_position("X", "BTC", qty, entry_price=price)
    trade = rs.get_trade("BTC")
    assert trade["stop"] == pytest.approx(expected)


def test_update_position_uses_atr_for_stop():
    guard = PortfolioGuard(GuardConfig(venue="test"))
    rs = RiskService(
        guard, account=Account(float("inf")), risk_pct=0.02, atr_mult=2.0, risk_per_trade=1.0
    )
    price = 100.0
    atr = 1.5
    rs.update_position("X", "BTC", 1.0, entry_price=price, atr=atr)
    trade = rs.get_trade("BTC")
    expected = price - 2.0 * atr
    assert trade["stop"] == pytest.approx(expected)
    assert trade["atr"] == pytest.approx(atr)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_on_fill_sets_initial_stop(side):
    risk_pct = 0.02
    price = 100.0
    expected = price * (1 - risk_pct) if side == "buy" else price * (1 + risk_pct)
    guard = PortfolioGuard(GuardConfig(venue="test"))
    rs = RiskService(
        guard, account=Account(float("inf")), risk_pct=risk_pct, risk_per_trade=1.0
    )
    rs.on_fill("BTC", side, 1.0, price=price, venue="X")
    trade = rs.get_trade("BTC")
    assert trade["stop"] == pytest.approx(expected)


def test_daily_loss_limit_via_guard():
    from datetime import datetime, timezone
    from tradingbot.risk.daily_guard import DailyGuard, GuardLimits

    guard = DailyGuard(GuardLimits(daily_max_loss_pct=0.05), venue="paper")
    now = datetime.now(timezone.utc)
    guard.on_mark(now, equity_now=1000.0)
    guard.on_realized_delta(-60)
    guard.on_mark(now, equity_now=940.0)
    halted, reason = guard.check_halt()
    assert halted and reason == "daily_loss"


@pytest.mark.asyncio
async def test_daily_guard_halts_on_loss():
    from datetime import datetime, timezone
    from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
    from tradingbot.execution.paper import PaperAdapter

    broker = PaperAdapter()
    symbol = "BTC/USDT"
    guard = DailyGuard(GuardLimits(daily_max_loss_pct=0.05), venue="paper")
    broker.state.cash = 100.0

    broker.update_last_price(symbol, 100.0)
    guard.on_mark(
        datetime.now(timezone.utc),
        equity_now=broker.equity(mark_prices={symbol: 100.0}),
    )
    buy = await broker.place_order(symbol, "buy", "limit", 1, price=100.0)

    broker.update_last_price(symbol, 90.0)
    guard.on_mark(
        datetime.now(timezone.utc),
        equity_now=broker.equity(mark_prices={symbol: 90.0}),
    )
    sell = await broker.place_order(symbol, "sell", "limit", 1, price=90.0)
    delta = (sell["price"] - buy["price"]) * 1
    guard.on_realized_delta(delta)
    halted, reason = guard.check_halt()
    assert halted and reason == "daily_loss"


def test_long_only_prevents_shorts():
    rs = _make_rs(100.0)
    rs.allow_short = False
    rs.add_fill("buy", 1.0, price=100.0)
    rs.update_position("test", "SYM", 1.0, entry_price=100.0)
    allowed, _, delta = rs.check_order("SYM", "sell", 100.0)
    assert allowed and delta == pytest.approx(-1.0)


def test_min_order_qty_blocks_small_orders():
    rs = _make_rs(100.0)
    rs.min_order_qty = 0.01
    allowed, reason, delta = rs.check_order("SYM", "buy", 100.0, strength=0.001)
    assert not allowed
    assert reason == "below_min_qty"
    assert delta == 0.0
