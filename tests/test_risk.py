import asyncio
import pytest


def test_size_scales_with_equity_and_strength():
    from tradingbot.risk.manager import RiskManager

    price = 100.0
    equity_small = 10_000.0
    equity_big = 20_000.0
    rm_small = RiskManager()
    rm_small.equity_pct = 1.0
    rm_big = RiskManager()
    rm_big.equity_pct = 1.0

    expected_small = equity_small * 0.5 / price
    expected_big = equity_big * 0.5 / price
    assert rm_small.size("buy", price, equity_small, strength=0.5) == pytest.approx(expected_small)
    assert rm_big.size("buy", price, equity_big, strength=0.5) == pytest.approx(expected_big)


def test_stop_loss_risk_pct():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.risk.exceptions import StopLossExceeded

    equity = 10_000.0
    risk_pct = 0.02
    price = 100.0

    qty = equity * 0.10 / price
    rm = RiskManager(risk_pct=risk_pct)
    rm.equity_pct = 1.0
    rm.set_position(qty)

    assert rm.check_limits(price)
    with pytest.raises(StopLossExceeded):
        rm.check_limits(price * (1 - risk_pct - 0.01))
    assert rm.enabled is True
    assert rm.last_kill_reason == "stop_loss"


def test_pyramiding_and_scaling(risk_manager):
    rm = risk_manager
    max_qty = rm.equity / rm.price

    delta = rm.size("buy", rm.price, rm.equity, strength=0.5)
    rm.add_fill("buy", delta)
    assert rm.pos.qty == pytest.approx(max_qty * 0.5)

    delta = rm.size("buy", rm.price, rm.equity, strength=1.0)
    rm.add_fill("buy", delta)
    assert rm.pos.qty == pytest.approx(max_qty)

    delta = rm.size("buy", rm.price, rm.equity, strength=0.5)
    rm.add_fill("sell", abs(delta))
    assert rm.pos.qty == pytest.approx(max_qty * 0.5)

    delta = rm.size("buy", rm.price, rm.equity, strength=0.0)
    rm.add_fill("sell", abs(delta))
    assert rm.pos.qty == pytest.approx(0.0)


def test_size_with_volatility_event():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager(vol_target=0.02)
    rm.equity_pct = 1.0
    before = RISK_EVENTS.labels(event_type="volatility_sizing")._value.get()
    delta = rm.size_with_volatility(0.04, price=1.0, equity=10)
    after = RISK_EVENTS.labels(event_type="volatility_sizing")._value.get()
    assert delta == pytest.approx(10.0)
    assert after == before + 1


def test_update_correlation_limits_exposure():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager()
    rm.equity_pct = 1.0
    pairs = {("BTC", "ETH"): 0.9}
    before = RISK_EVENTS.labels(event_type="correlation_limit")._value.get()
    exceeded = rm.update_correlation(pairs, 0.8)
    after = RISK_EVENTS.labels(event_type="correlation_limit")._value.get()
    assert exceeded == [("BTC", "ETH")]
    assert after == before + 1


def test_kill_switch_disables():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager()
    rm.equity_pct = 1.0
    before = RISK_EVENTS.labels(event_type="kill_switch")._value.get()
    rm.kill_switch("manual")
    after = RISK_EVENTS.labels(event_type="kill_switch")._value.get()
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert after == before + 1


@pytest.mark.asyncio
async def test_daily_loss_limit_via_bus():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.bus import EventBus

    bus = EventBus()
    events = []
    bus.subscribe("risk:halted", lambda e: events.append(e))
    rm = RiskManager(daily_loss_limit=50, bus=bus)
    rm.equity_pct = 1.0
    await bus.publish("pnl", {"delta": -60})
    await asyncio.sleep(0)
    assert rm.enabled is False
    assert events and events[0]["reason"] == "daily_loss"


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
    guard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: 100.0}))
    buy = await broker.place_order(symbol, "buy", "market", 1)

    broker.update_last_price(symbol, 90.0)
    guard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: 90.0}))
    sell = await broker.place_order(symbol, "sell", "market", 1)
    delta = (sell["price"] - buy["price"]) * 1
    guard.on_realized_delta(delta)
    halted, reason = guard.check_halt()
    assert halted and reason == "daily_loss"


def test_covariance_limit_triggers_kill():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager()
    rm.equity_pct = 1.0
    positions = {"BTC": 1.0, "ETH": 1.0}
    cov = {
        ("BTC", "BTC"): 0.04,
        ("ETH", "ETH"): 0.04,
        ("BTC", "ETH"): 0.039,
    }
    ok = rm.check_portfolio_risk(positions, cov, max_variance=0.1)
    assert ok is False
    assert rm.enabled is False
    assert rm.last_kill_reason == "covariance_limit"
