import pytest
from datetime import datetime, timezone

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.bus import EventBus
from tradingbot.data.ingest import run_orderbook_stream
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

    def fake_insert(engine, **data):
        inserted.append(data)

    monkeypatch.setattr("tradingbot.data.ingest.insert_orderbook", fake_insert)

    await run_orderbook_stream(adapter, "BTC/USDT", depth=5, bus=bus, engine="engine")

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
