import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.bus import EventBus
from tradingbot.data.ingestion import run_orderbook_stream, run_trades_stream
from tradingbot.data import ingestion
from tradingbot.types import OrderBook


class DummyOBAdapter(ExchangeAdapter):
    name = "dummy"

    def __init__(self, snapshots):
        self._snapshots = snapshots

    async def stream_trades(self, symbol: str):
        if False:
            yield {}

    async def stream_orderbook(self, symbol: str, depth: int):
        for snap in self._snapshots:
            yield snap

    async def place_order(self, *args, **kwargs):
        return {}

    async def cancel_order(self, order_id: str):
        return {}


@pytest.mark.asyncio
async def test_run_orderbook_stream_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    snapshot = {
        "ts": ts,
        "bid_px": [100.0, 99.5],
        "bid_qty": [1.0, 2.0],
        "ask_px": [100.5, 101.0],
        "ask_qty": [1.5, 2.5],
    }
    adapter = DummyOBAdapter([snapshot])
    bus = EventBus()
    published = []
    bus.subscribe("orderbook", lambda ob: published.append(ob))

    inserted = []

    class DummyStorage:
        def insert_orderbook(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    await run_orderbook_stream(
        adapter, "BTC/USDT", depth=5, bus=bus, engine="engine"
    )

    assert len(published) == 1
    ob = published[0]
    assert isinstance(ob, OrderBook)
    assert ob.bid_px == [100.0, 99.5]
    assert len(inserted) == 1
    assert inserted[0]["symbol"] == "BTC/USDT"
    assert inserted[0]["bid_px"] == [100.0, 99.5]


@pytest.mark.asyncio
async def test_stream_orderbook_persists_levels(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    snapshot = {
        "ts": ts,
        "bid_px": [100.0, 99.5],
        "bid_qty": [1.0, 2.0],
        "ask_px": [100.5, 101.0],
        "ask_qty": [1.5, 2.5],
    }

    class DummyAdapter(ExchangeAdapter):
        name = "dummy"

        async def stream_trades(self, symbol: str):
            if False:
                yield {}

        async def stream_orderbook(self, symbol: str, depth: int):
            yield snapshot

        async def place_order(self, *args, **kwargs):
            return {}

        async def cancel_order(self, order_id: str):
            return {}

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_orderbook(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    await ingestion.stream_orderbook(DummyAdapter(), "BTC/USDT", depth=5)

    assert len(inserted) == 1
    assert inserted[0]["bid_px"] == [100.0, 99.5]
    assert inserted[0]["ask_qty"] == [1.5, 2.5]


@pytest.mark.asyncio
async def test_run_trades_stream_publishes():
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter(ExchangeAdapter):
        name = "dummy"

        async def stream_trades(self, symbol: str):
            yield {"ts": ts, "price": 1.0, "qty": 2.0, "side": "buy"}

        async def stream_orderbook(self, symbol: str, depth: int):
            if False:
                yield {}

        async def place_order(self, *args, **kwargs):
            return {}

        async def cancel_order(self, order_id: str):
            return {}

    bus = EventBus()
    published = []
    bus.subscribe("trades", lambda tick: published.append(tick))

    await run_trades_stream(DummyAdapter(), "BTC/USDT", bus)

    assert len(published) == 1
    tick = published[0]
    assert tick.price == 1.0
    assert tick.qty == 2.0


@pytest.mark.asyncio
async def test_poll_funding_inserts(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_funding(self, symbol: str):
            return {"ts": ts, "rate": 0.01, "interval_sec": 60}

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_funding(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    task = asyncio.create_task(
        ingestion.poll_funding(DummyAdapter(), "BTC/USDT", interval=0)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inserted
    assert inserted[0]["rate"] == 0.01


@pytest.mark.asyncio
async def test_poll_open_interest_inserts(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_oi(self, symbol: str):
            return {"ts": ts, "oi": 123.0}

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_open_interest(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    task = asyncio.create_task(
        ingestion.poll_open_interest(DummyAdapter(), "BTC/USDT", interval=0)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inserted
    assert inserted[0]["oi"] == 123.0
