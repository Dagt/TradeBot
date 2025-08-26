
import asyncio
import pytest

def test_risk_manager_position(risk_manager):
    risk_manager.set_position(1)
    risk_manager.add_fill("buy", 2)
    assert risk_manager.pos.qty == 3
    assert risk_manager.size("buy") == pytest.approx(-2)


def test_stop_loss_triggers_disable():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(stop_loss_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    # caída del 6% -> excede stop_loss_pct
    assert not rm.check_limits(94)
    assert rm.enabled is False


def test_drawdown_triggers_disable():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(max_drawdown_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert rm.check_limits(110)  # peak
    # retroceso >5% desde el pico
    assert not rm.check_limits(104)
    assert rm.enabled is False


def test_trailing_stop_tracks_peak():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(max_drawdown_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert rm.check_limits(105)
    # retroceso menor al 5% desde el nuevo pico -> sigue habilitado
    assert rm.check_limits(101)
    assert rm.enabled is True


def test_size_with_volatility_event():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager(vol_target=0.02)
    before = RISK_EVENTS.labels(event_type="volatility_sizing")._value.get()
    delta = rm.size_with_volatility(0.04)
    after = RISK_EVENTS.labels(event_type="volatility_sizing")._value.get()
    assert delta == pytest.approx(0.5)
    assert after == before + 1


def test_update_correlation_limits_exposure():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager()
    pairs = {("BTC", "ETH"): 0.9}
    before = RISK_EVENTS.labels(event_type="correlation_limit")._value.get()
    exceeded = rm.update_correlation(pairs, 0.8)
    after = RISK_EVENTS.labels(event_type="correlation_limit")._value.get()
    assert exceeded == [("BTC", "ETH")]
    assert rm.max_pos == pytest.approx(0.5)
    assert after == before + 1


def test_kill_switch_disables():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager()
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
    guard = DailyGuard(GuardLimits(daily_max_loss_usdt=5.0), venue="paper")

    broker.update_last_price(symbol, 100.0)
    guard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: 100.0}))
    buy = await broker.place_order(symbol, "buy", "market", 1)

    broker.update_last_price(symbol, 90.0)
    guard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: 90.0}))
    sell = await broker.place_order(symbol, "sell", "market", 1)
    delta = (sell["price"] - buy["price"]) * 1
    guard.on_realized_delta(delta)
    halted, reason = guard.check_halt()
    assert halted and reason == "daily_max_loss"


def test_covariance_limit_triggers_kill():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager()
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
