import asyncio
import pytest
import asyncio
import pytest

from tradingbot.live.daemon import TradeBotDaemon
from tradingbot.risk.manager import RiskManager
from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.paper import PaperAdapter
from tradingbot.strategies.base import Signal
from tradingbot.bus import EventBus


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
    risk = RiskManager(max_pos=5, bus=bus)
    daemon = TradeBotDaemon({"feed": adapter}, [AlwaysBuy()], risk, router, ["BTCUSDT"])
    task = asyncio.create_task(daemon.run())
    await asyncio.sleep(0.3)
    daemon._stop.set()
    await task
    assert risk.pos.qty > 0
