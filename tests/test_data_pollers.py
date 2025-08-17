import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.bus import EventBus
from tradingbot.data import funding as data_funding
from tradingbot.data import open_interest as data_oi
from tradingbot.connectors.base import Funding, OpenInterest


class DummyConnector:
    """Connector exposing funding and open interest methods."""

    name = "dummy"

    async def fetch_funding(self, symbol: str):
        return Funding(
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            exchange=self.name,
            symbol=symbol,
            rate=0.01,
        )

    async def fetch_open_interest(self, symbol: str):
        return OpenInterest(
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            exchange=self.name,
            symbol=symbol,
            oi=123.0,
        )


@pytest.mark.asyncio
async def test_poll_funding_publishes_event():
    bus = EventBus()
    events = []
    bus.subscribe("funding", lambda e: events.append(e))

    task = asyncio.create_task(
        data_funding.poll_funding(DummyConnector(), "BTC/USDT", bus, interval=0.1)
    )
    await asyncio.sleep(0.11)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events
    assert events[0]["rate"] == 0.01
    assert events[0]["exchange"] == "dummy"


@pytest.mark.asyncio
async def test_poll_open_interest_publishes_event():
    bus = EventBus()
    events = []
    bus.subscribe("open_interest", lambda e: events.append(e))

    task = asyncio.create_task(
        data_oi.poll_open_interest(DummyConnector(), "BTC/USDT", bus, interval=0.1)
    )
    await asyncio.sleep(0.11)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events
    assert events[0]["oi"] == 123.0
    assert events[0]["exchange"] == "dummy"
