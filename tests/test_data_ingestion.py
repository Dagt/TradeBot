import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.data.ingestion import (
    poll_funding,
    poll_open_interest,
    stream_orderbook,
    stream_trades,
)
from tradingbot.types import Tick


class DummyTradesAdapter:
    name = "dummy"

    def __init__(self, trades):
        self._trades = trades

    async def stream_trades(self, symbol: str):
        for t in self._trades:
            yield t

    async def stream_orderbook(self, symbol: str, depth: int):
        if False:
            yield {}


class DummyOBAdapter:
    name = "dummy"

    def __init__(self, snapshots):
        self._snapshots = snapshots

    async def stream_trades(self, symbol: str):
        if False:
            yield {}

    async def stream_orderbook(self, symbol: str, depth: int):
        for snap in self._snapshots:
            yield snap


class DummyInfoAdapter:
    name = "dummy"

    def __init__(self, info):
        self._info = info

    async def fetch_funding(self, symbol: str):
        return self._info

    async def fetch_oi(self, symbol: str):
        return self._info


@pytest.mark.asyncio
async def test_stream_trades_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    trade = {"ts": ts, "price": 100.0, "qty": 1.0, "side": "buy"}
    adapter = DummyTradesAdapter([trade])

    inserted = []

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.get_engine", lambda: "engine"
    )

    def fake_insert(engine, tick: Tick):
        inserted.append(tick)

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.insert_trade", fake_insert
    )

    await stream_trades(adapter, "BTC/USDT")

    assert len(inserted) == 1
    assert inserted[0].price == 100.0


@pytest.mark.asyncio
async def test_stream_orderbook_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    snapshot = {
        "ts": ts,
        "bid_px": [100.0, 99.5],
        "bid_qty": [1.0, 2.0],
        "ask_px": [100.5, 101.0],
        "ask_qty": [1.5, 2.5],
    }
    adapter = DummyOBAdapter([snapshot])

    inserted = []

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.get_engine", lambda: "engine"
    )

    def fake_insert(engine, **data):
        inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.insert_orderbook", fake_insert
    )

    await stream_orderbook(adapter, "BTC/USDT", depth=5)

    assert len(inserted) == 1
    assert inserted[0]["symbol"] == "BTC/USDT"


@pytest.mark.asyncio
async def test_poll_funding_once(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    info = {"ts": ts, "rate": 0.01, "interval_sec": 8}
    adapter = DummyInfoAdapter(info)

    inserted = []

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.get_engine", lambda: "engine"
    )

    def fake_insert(engine, **data):
        inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.insert_funding", fake_insert
    )

    async def fake_sleep(_):
        raise asyncio.CancelledError

    monkeypatch.setattr("tradingbot.data.ingestion.asyncio.sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await poll_funding(adapter, "BTC/USDT", interval=0)

    assert len(inserted) == 1
    assert inserted[0]["rate"] == 0.01


@pytest.mark.asyncio
async def test_poll_open_interest_once(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    info = {"ts": ts, "oi": 123.0}
    adapter = DummyInfoAdapter(info)

    inserted = []

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.get_engine", lambda: "engine"
    )

    def fake_insert(engine, **data):
        inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion.ts_storage.insert_open_interest", fake_insert
    )

    async def fake_sleep(_):
        raise asyncio.CancelledError

    monkeypatch.setattr("tradingbot.data.ingestion.asyncio.sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await poll_open_interest(adapter, "BTC/USDT", interval=0)

    assert len(inserted) == 1
    assert inserted[0]["oi"] == 123.0

