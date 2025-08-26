import asyncio
import pytest
import asyncio

from tradingbot.live.daemon import TradeBotDaemon
from tradingbot.risk.manager import RiskManager
from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.paper import PaperAdapter
from tradingbot.strategies.base import Signal
from tradingbot.bus import EventBus
from collections import deque


class FeedAdapter:
    def __init__(self, trades):
        self._trades = trades
        class Rest:
            async def fetch_balance(self_inner):
                return {}
        self.rest = Rest()

    async def stream_trades(self, symbol):
        for t in self._trades:
            yield {"symbol": symbol, **t}
            await asyncio.sleep(0)


class AlwaysBuy:
    name = "always_buy"

    def on_trade(self, trade):
        return Signal("buy", 1.0)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_daemon_processes_trades():
    trades = [{"price": 100.0, "qty": 1.0, "side": "buy"}, {"price": 101.0, "qty": 1.0, "side": "buy"}]
    adapter = FeedAdapter(trades)
    paper = PaperAdapter()
    paper.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(paper)
    bus = EventBus()
    risk = RiskManager(bus=bus)
    daemon = TradeBotDaemon({"feed": adapter}, [AlwaysBuy()], risk, router, ["BTCUSDT"])
    task = asyncio.create_task(daemon.run())
    await asyncio.sleep(0.3)
    daemon._stop.set()
    await task
    assert risk.pos.qty > 0


@pytest.mark.asyncio
async def test_daemon_adjusts_size_for_correlation():
    class DummyRisk(RiskManager):
        def update_correlation(self, *args, **kwargs):
            return []

        def update_covariance(self, *args, **kwargs):
            return []

    class DummyRouter:
        def __init__(self):
            self.orders = []

        async def execute(self, order):
            self.orders.append(order)
            return {"status": "filled"}

    risk = DummyRisk()
    router = DummyRouter()
    daemon = TradeBotDaemon({}, [], risk, router, ["AAA"], returns_window=10)
    daemon.price_history["AAA"] = deque([1, 2, 3], maxlen=10)
    daemon.price_history["BBB"] = deque([2, 4, 6, 8], maxlen=10)

    trade = type("T", (), {"symbol": "AAA", "price": 4.0})
    sig = Signal("buy", 1.0)

    await daemon._on_signal({"signal": sig, "trade": trade})

    assert router.orders and router.orders[0].qty == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_daemon_emits_event_on_high_correlation():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:paused", lambda e: events.append(e))
    risk = RiskManager(bus=bus)
    router = ExecutionRouter(PaperAdapter())
    daemon = TradeBotDaemon({}, [], risk, router, ["AAA", "BBB"], returns_window=5)
    daemon.price_history["AAA"] = deque([1, 2, 3], maxlen=5)
    daemon.price_history["BBB"] = deque([2, 4, 6], maxlen=5)
    trade = type("T", (), {"symbol": "AAA", "price": 4.0})
    sig = Signal("buy", 1.0)
    await daemon._on_signal({"signal": sig, "trade": trade})
    await asyncio.sleep(0)
    assert risk.max_pos == pytest.approx(0.5)
    assert events and events[0]["reason"] == "correlation"
