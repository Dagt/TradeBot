import asyncio
import pytest

from tradingbot.bus import EventBus
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
from tradingbot.risk.service import RiskService
from tradingbot.storage import timescale


def test_stop_loss_sets_reason():
    rm = RiskManager(max_pos=1, stop_loss_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert not rm.check_limits(94)
    assert rm.enabled is False
    assert rm.last_kill_reason == "stop_loss"
    assert rm.pos.qty == 0


def test_drawdown_sets_reason():
    rm = RiskManager(max_pos=1, max_drawdown_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert rm.check_limits(110)
    assert not rm.check_limits(104)
    assert rm.enabled is False
    assert rm.last_kill_reason == "drawdown"
    assert rm.pos.qty == 0


def test_manual_kill_switch_records_reason():
    rm = RiskManager(max_pos=1)
    rm.kill_switch("manual")
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert rm.pos.qty == 0


def test_daily_loss_limit_triggers_kill_switch():
    rm = RiskManager(max_pos=1, daily_loss_limit=50)
    rm.set_position(1)
    rm.check_limits(100)
    rm.update_pnl(-60)
    # segundo check_limits evalúa límites diarios
    assert not rm.check_limits(100)
    assert rm.enabled is False
    assert rm.last_kill_reason == "daily_loss"
    assert rm.pos.qty == 0


def test_risk_service_updates_and_persists(monkeypatch):
    rm = RiskManager()
    guard = PortfolioGuard(GuardConfig(total_cap_usdt=1.0, per_symbol_cap_usdt=1.0, venue="X"))
    daily = DailyGuard(GuardLimits(), venue="X")
    events: list = []
    monkeypatch.setattr(timescale, "insert_risk_event", lambda engine, **kw: events.append(kw))
    svc = RiskService(rm, guard, daily, engine=object())
    svc.update_position("ex1", "BTC", 1.0)
    svc.update_position("ex2", "BTC", -0.4)
    agg = svc.aggregate_positions()
    assert agg["BTC"] == pytest.approx(0.6)
    allowed, _, _delta = svc.check_order("BTC", "buy", 1.0, strength=2.0)
    assert not allowed
    assert events and events[0]["kind"] == "VIOLATION"


@pytest.mark.asyncio
async def test_update_correlation_emits_pause():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:paused", lambda e: events.append(e))
    rm = RiskManager(max_pos=8, bus=bus)
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
    rm = RiskManager(max_pos=8, bus=bus)
    cov = {
        ("BTC", "BTC"): 0.04,
        ("ETH", "ETH"): 0.04,
        ("BTC", "ETH"): 0.039,
    }
    exceeded = rm.update_covariance(cov, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("BTC", "ETH")]
    assert events and events[0]["reason"] == "covariance"

