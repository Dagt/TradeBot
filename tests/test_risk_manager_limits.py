import asyncio
import pytest
from tradingbot.bus import EventBus
from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
from tradingbot.risk.service import RiskService
from tradingbot.risk.limits import RiskLimits, LimitTracker
from tradingbot.storage import timescale


def _make_service(risk_pct: float = 0.05) -> RiskService:
    guard = PortfolioGuard(GuardConfig(venue="X"))
    return RiskService(guard, account=Account(float("inf")), risk_pct=risk_pct)


def test_stop_loss_sets_reason():
    from tradingbot.risk.exceptions import StopLossExceeded

    svc = _make_service(risk_pct=0.05)
    svc.rm.add_fill("buy", 1.0, price=100.0)
    assert svc.rm.check_limits(100.0)
    with pytest.raises(StopLossExceeded):
        svc.rm.check_limits(94.0)
    assert svc.rm.enabled is True
    assert svc.rm.last_kill_reason == "stop_loss"
    assert svc.rm.pos.qty == pytest.approx(1.0)


def test_stop_loss_multiple_fills_weighted_average():
    from tradingbot.risk.exceptions import StopLossExceeded

    svc = _make_service(risk_pct=0.1)
    svc.rm.add_fill("buy", 1, price=100)
    svc.rm.add_fill("buy", 1, price=120)
    assert svc.rm._entry_price == pytest.approx(110.0)
    assert svc.rm.pos.qty == pytest.approx(2.0)
    assert svc.rm.check_limits(105)
    with pytest.raises(StopLossExceeded):
        svc.rm.check_limits(98)


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
    svc = RiskService(
        guard,
        account=Account(float("inf")),
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.05,
    )
    svc.account.cash = 100.0
    svc.rm.add_fill("buy", 1.0, price=100.0)
    svc.update_position("X", "BTC", 1.0, entry_price=100.0)
    allowed, reason, delta = svc.check_order("BTC", "buy", 94.0)
    assert allowed is True
    assert reason == "stop_loss"
    assert delta == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_update_correlation_returns_pairs():
    svc = _make_service()
    pairs = {("BTC", "ETH"): 0.9}
    exceeded = svc.rm.update_correlation(pairs, 0.8)
    assert exceeded == [("BTC", "ETH")]


@pytest.mark.asyncio
async def test_update_covariance_returns_pairs():
    svc = _make_service()
    cov = {
        ("BTC", "BTC"): 0.04,
        ("ETH", "ETH"): 0.04,
        ("BTC", "ETH"): 0.039,
    }
    exceeded = svc.rm.update_covariance(cov, 0.8)
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
    assert tracker.blocked


@pytest.mark.asyncio
async def test_hard_pnl_stop_blocks():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:blocked", lambda e: events.append(e))
    tracker = LimitTracker(RiskLimits(hard_pnl_stop=100), bus=bus)
    tracker.update_pnl(-120)
    await asyncio.sleep(0)
    assert events and events[0]["reason"] == "hard_pnl_stop"
    assert tracker.blocked
