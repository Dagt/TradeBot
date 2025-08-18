import asyncio

import pytest

from tradingbot.bus import EventBus
from tradingbot.live.daemon import TradeBotDaemon
from tradingbot.execution.router import ExecutionRouter
from tradingbot.risk.manager import RiskManager


class DummyAdapter:
    name = "dummy"

    async def stream_trades(self, symbol: str):
        if False:
            yield {}


class DummyStrategy:
    def __init__(self):
        self.funding = []
        self.basis = []

    def on_funding(self, evt):
        self.funding.append(evt)

    def on_basis(self, evt):
        self.basis.append(evt)


@pytest.mark.asyncio
async def test_daemon_routes_funding_and_basis():
    bus = EventBus()
    risk = RiskManager(bus=bus)
    router = ExecutionRouter([])
    strat = DummyStrategy()
    daemon = TradeBotDaemon({"d": DummyAdapter()}, [strat], risk, router, symbols=["BTCUSDT"])

    await bus.publish("funding", {"rate": 1})
    await bus.publish("basis", {"basis": 2})

    assert strat.funding and strat.funding[0]["rate"] == 1
    assert strat.basis and strat.basis[0]["basis"] == 2
