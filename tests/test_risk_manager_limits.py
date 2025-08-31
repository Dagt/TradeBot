import asyncio
import pytest

from tradingbot.bus import EventBus
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
from tradingbot.risk.service import RiskService
from tradingbot.storage import timescale
from tradingbot.utils.metrics import KILL_SWITCH_ACTIVE
from tradingbot.risk.limits import RiskLimits, LimitTracker
from tradingbot.core import Account


def test_stop_loss_sets_reason():
    from tradingbot.risk.exceptions import StopLossExceeded

    svc = RiskService(
        PortfolioGuard(GuardConfig(venue="X")), account=Account(float("inf")), risk_pct=0.05
    )
    rm = svc.rm
    rm.add_fill("buy", 1, price=100)
    assert rm.check_limits(100)
    with pytest.raises(StopLossExceeded):
        rm.check_limits(94)
    assert rm.enabled is True
    assert rm.last_kill_reason == "stop_loss"
    assert rm.pos.qty == pytest.approx(1.0)


def test_stop_loss_multiple_fills_weighted_average():
    from tradingbot.risk.exceptions import StopLossExceeded

    svc = RiskService(
        PortfolioGuard(GuardConfig(venue="X")), account=Account(float("inf")), risk_pct=0.1
    )
    rm = svc.rm
    rm.add_fill("buy", 1, price=100)
    rm.add_fill("buy", 1, price=120)
    # Precio de entrada promedio = 110
    assert rm._entry_price == pytest.approx(110.0)
    assert rm.pos.qty == pytest.approx(2.0)

    # Precio no dispara el stop todavía
    assert rm.check_limits(105)

    # Caída por debajo del umbral de stop-loss
    with pytest.raises(StopLossExceeded):
        rm.check_limits(98)


def test_manual_kill_switch_records_reason():
    rm = RiskService(PortfolioGuard(GuardConfig(venue="X")), account=Account(float("inf"))).rm
    rm.enabled = False
    rm.last_kill_reason = "manual"
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert rm.pos.qty == 0


def test_reset_clears_kill_switch():
    rm = RiskService(PortfolioGuard(GuardConfig(venue="X")), account=Account(float("inf"))).rm
    rm.enabled = False
    rm.last_kill_reason = "manual"
    KILL_SWITCH_ACTIVE._value.set(1.0)
    assert rm.enabled is False
    rm.reset()
    assert rm.enabled is True
    assert rm.last_kill_reason is None
    assert rm.pos.qty == 0
    assert KILL_SWITCH_ACTIVE._value.get() == 0.0


def test_daily_loss_limit_triggers_kill_switch():
    tracker = LimitTracker(RiskLimits(hard_pnl_stop=50))
    tracker.update_pnl(-60)
    assert tracker.blocked is True
    assert tracker.last_block_reason == "hard_pnl_stop"


def test_risk_service_updates_and_persists(monkeypatch):
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=0.5, per_symbol_cap_pct=0.5, venue="X")
    )
    guard.refresh_usd_caps(1.0)
    daily = DailyGuard(GuardLimits(), venue="X")
    events: list = []
    monkeypatch.setattr(
        timescale, "insert_risk_event", lambda engine, **kw: events.append(kw)
    )
    account = Account(float("inf"))
    svc = RiskService(
        guard,
        daily,
        engine=object(),
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=1.0,
    )
    svc.account.cash = 100.0
    svc.rm.enabled = False
    svc.rm.last_kill_reason = "manual"
    allowed, _, _delta = svc.check_order("BTC", "buy", 1.0, strength=1.0)
    assert not allowed
    assert events and events[0]["kind"] == "VIOLATION"


def test_risk_service_stop_loss_triggers_close():
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    account = Account(float("inf"))
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.05,
    )
    svc.rm.add_fill("buy", 1.0, price=100.0)
    svc.update_position("X", "BTC", 1.0, entry_price=100.0)
    svc.rm.check_limits(100.0)

    allowed, reason, delta = svc.check_order("BTC", "buy", 94.0)
    assert allowed is True
    assert reason == "stop_loss"
    assert delta == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_update_correlation_emits_pause():
    rm = RiskService(
        PortfolioGuard(GuardConfig(venue="X")), account=Account(float("inf"))
    ).rm
    pairs = {("BTC", "ETH"): 0.9}
    exceeded = rm.update_correlation(pairs, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("BTC", "ETH")]


@pytest.mark.asyncio
async def test_update_covariance_emits_pause():
    rm = RiskService(
        PortfolioGuard(GuardConfig(venue="X")), account=Account(float("inf"))
    ).rm
    cov = {
        ("BTC", "BTC"): 0.04,
        ("ETH", "ETH"): 0.04,
        ("BTC", "ETH"): 0.039,
    }
    exceeded = rm.update_covariance(cov, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("BTC", "ETH")]


def test_register_order_notional_limit():
    tracker = LimitTracker(RiskLimits(max_notional=100))
    assert tracker.register_order(50)
    assert not tracker.register_order(150)


def test_concurrent_order_limit():
    tracker = LimitTracker(RiskLimits(max_concurrent_orders=1))
    assert tracker.register_order(10)
    assert not tracker.register_order(10)
    tracker.complete_order()
    assert tracker.register_order(10)


@pytest.mark.asyncio
async def test_daily_dd_limit_blocks_and_emits_event():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:blocked", lambda e: events.append(e))
    tracker = LimitTracker(RiskLimits(daily_dd_limit=50), bus=bus)
    tracker.update_pnl(100)
    tracker.update_pnl(-160)
    await asyncio.sleep(0)
    assert events and events[0]["reason"] == "daily_dd_limit"
    assert tracker.blocked is True


@pytest.mark.asyncio
async def test_hard_pnl_stop_blocks():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:blocked", lambda e: events.append(e))
    tracker = LimitTracker(RiskLimits(hard_pnl_stop=100), bus=bus)
    tracker.update_pnl(-120)
    await asyncio.sleep(0)
    assert events and events[0]["reason"] == "hard_pnl_stop"
    assert tracker.blocked is True

