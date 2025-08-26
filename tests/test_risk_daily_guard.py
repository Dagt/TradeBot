import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.risk.daily_guard import DailyGuard, GuardLimits
from tradingbot.execution.paper import PaperAdapter
from tradingbot.storage import timescale
from tradingbot.bus import EventBus
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.limits import RiskLimits


@pytest.mark.asyncio
async def test_daily_guard_close_and_persist(monkeypatch):
    broker = PaperAdapter()
    broker.state.cash = 1000.0
    symbol = "BTC/USDT"
    broker.update_last_price(symbol, 100.0)

    guard = DailyGuard(
        GuardLimits(daily_max_loss_pct=1.0, daily_max_drawdown_pct=0.05, halt_action="close_all"),
        venue="paper",
        storage_engine="eng",
    )

    # initialize day with current equity
    guard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: 100.0}))
    await broker.place_order(symbol, "buy", "market", 1)

    broker.update_last_price(symbol, 50.0)
    guard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: 50.0}))

    calls = []

    def fake_insert(engine, venue, symbol, kind, message, details=None):
        calls.append((engine, venue, symbol, kind, message, details))

    monkeypatch.setattr(timescale, "insert_risk_event", fake_insert)

    halted, reason = guard.check_halt(broker)
    await asyncio.sleep(0)

    assert halted and reason == "daily_drawdown"
    assert broker.state.pos[symbol].qty == 0
    assert calls and calls[0][3] == "daily_drawdown"
    assert calls[0][0] == "eng"


@pytest.mark.asyncio
async def test_daily_dd_limit_blocks_risk_manager():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:blocked", lambda e: events.append(e))

    rm = RiskManager(bus=bus, limits=RiskLimits(daily_dd_limit=50))
    rm.set_position(1.0)

    rm.update_pnl(100)
    rm.update_pnl(-160)

    await asyncio.sleep(0)

    assert events and events[0]["reason"] == "daily_dd_limit"
    assert rm.enabled is False
    assert rm.pos.qty == 0.0
    assert rm.limits and rm.limits.blocked
