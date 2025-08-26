import asyncio
import pytest

from tradingbot.bus import EventBus
from tradingbot.risk.manager import RiskManager, RiskPctViolation
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
from tradingbot.risk.service import RiskService
from tradingbot.storage import timescale
import asyncio
import pytest
from tradingbot.bus import EventBus
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
from tradingbot.risk.service import RiskService
from tradingbot.storage import timescale
from tradingbot.utils.metrics import KILL_SWITCH_ACTIVE
from tradingbot.risk.limits import RiskLimits


def test_stop_loss_triggers_exception():
    rm = RiskManager(equity_pct=1.0, risk_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    with pytest.raises(RiskPctViolation):
        rm.check_limits(94)
    assert rm.enabled is True
    assert rm.last_kill_reason is None
    assert rm.pos.qty == 1


def test_manual_kill_switch_records_reason():
    rm = RiskManager(equity_pct=1.0)
    rm.kill_switch("manual")
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert rm.pos.qty == 0


def test_reset_clears_kill_switch():
    rm = RiskManager(equity_pct=1.0)
    rm.kill_switch("manual")
    assert rm.enabled is False
    assert KILL_SWITCH_ACTIVE._value.get() == 1.0
    rm.reset()
    assert rm.enabled is True
    assert rm.last_kill_reason is None
    assert rm.pos.qty == 0
    assert KILL_SWITCH_ACTIVE._value.get() == 0.0


def test_daily_loss_limit_triggers_kill_switch():
    rm = RiskManager(equity_pct=1.0, daily_loss_limit=50)
    rm.set_position(1)
    rm.check_limits(100)
    rm.update_pnl(-60)
    # segundo check_limits evalúa límites diarios
    assert not rm.check_limits(100)
    assert rm.enabled is False
    assert rm.last_kill_reason == "daily_loss"
    assert rm.pos.qty == 0


def test_risk_service_updates_and_persists(monkeypatch):
    rm = RiskManager(equity_pct=1.0)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=0.5, per_symbol_cap_pct=0.5, venue="X")
    )
    guard.refresh_usd_caps(1.0)
    daily = DailyGuard(GuardLimits(), venue="X")
    events: list = []
    monkeypatch.setattr(
        timescale, "insert_risk_event", lambda engine, **kw: events.append(kw)
    )
    svc = RiskService(rm, guard, daily, engine=object())
    allowed, _, _delta = svc.check_order("BTC", "buy", 1.0, 1.0, strength=1.0)
    assert not allowed
    assert events and events[0]["kind"] == "VIOLATION"


def test_risk_service_close_signal_on_risk_pct():
    rm = RiskManager(equity_pct=1.0, risk_pct=0.05)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X")
    )
    svc = RiskService(rm, guard)
    rm.set_position(1)
    rm.check_limits(100)
    allowed, reason, delta = svc.check_order("BTC", "buy", 1.0, 94.0)
    assert not allowed
    assert reason == "risk_pct"
    assert delta == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_update_correlation_emits_pause():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:paused", lambda e: events.append(e))
    rm = RiskManager(equity_pct=1.0, bus=bus)
    pairs = {("BTC", "ETH"): 0.9}
    exceeded = rm.update_correlation(pairs, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("BTC", "ETH")]
    assert events and events[0]["reason"] == "correlation"


@pytest.mark.asyncio
async def test_update_covariance_emits_pause():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:paused", lambda e: events.append(e))
    rm = RiskManager(equity_pct=1.0, bus=bus)
    cov = {
        ("BTC", "BTC"): 0.04,
        ("ETH", "ETH"): 0.04,
        ("BTC", "ETH"): 0.039,
    }
    exceeded = rm.update_covariance(cov, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("BTC", "ETH")]
    assert events and events[0]["reason"] == "covariance"


def test_register_order_notional_limit():
    rm = RiskManager(limits=RiskLimits(max_notional=100))
    assert rm.register_order(50)
    assert not rm.register_order(150)


def test_concurrent_order_limit():
    rm = RiskManager(limits=RiskLimits(max_concurrent_orders=1))
    assert rm.register_order(10)
    assert not rm.register_order(10)
    rm.complete_order()
    assert rm.register_order(10)


@pytest.mark.asyncio
async def test_daily_dd_limit_blocks_and_emits_event():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:blocked", lambda e: events.append(e))
    rm = RiskManager(bus=bus, limits=RiskLimits(daily_dd_limit=50))
    rm.update_pnl(100)
    rm.update_pnl(-160)
    await asyncio.sleep(0)
    assert events and events[0]["reason"] == "daily_dd_limit"
    assert not rm.check_limits(100)


@pytest.mark.asyncio
async def test_hard_pnl_stop_blocks():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:blocked", lambda e: events.append(e))
    rm = RiskManager(bus=bus, limits=RiskLimits(hard_pnl_stop=100))
    rm.update_pnl(-120)
    await asyncio.sleep(0)
    assert events and events[0]["reason"] == "hard_pnl_stop"
    assert not rm.check_limits(100)

