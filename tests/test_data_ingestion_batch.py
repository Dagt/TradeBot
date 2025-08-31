from datetime import datetime, timezone

import pytest

from tradingbot.workers import OrderBookBatchWorker, run_orderbook_ingestion
from tradingbot.data.ingestion import (
    download_coinapi_open_interest,
    download_kaiko_open_interest,
    download_funding,
    download_kaiko_funding,
)


class DummyAdapter:
    def __init__(self, snaps):
        self._snaps = snaps
        self.name = "dummy"

    async def stream_order_book(self, symbol, depth):
        for snap in self._snaps:
            yield snap


@pytest.mark.asyncio
async def test_orderbook_batch_worker_flushes():
    inserted = []

    class DummyStorage:
        def insert_orderbook(self, engine, **data):
            inserted.append(data)

    storage = DummyStorage()
    worker = OrderBookBatchWorker(storage, engine="e", batch_size=2)

    snapshot = {
        "ts": datetime(2023, 1, 1, tzinfo=timezone.utc),
        "exchange": "ex",
        "symbol": "BTC/USDT",
        "bid_px": [1.0],
        "bid_qty": [2.0],
        "ask_px": [3.0],
        "ask_qty": [4.0],
    }

    await worker.add(snapshot)
    assert inserted == []

    await worker.add(snapshot)
    assert len(inserted) == 2

    await worker.add(snapshot)
    assert len(inserted) == 2
    await worker.flush()
    assert len(inserted) == 3


@pytest.mark.asyncio
async def test_run_orderbook_ingestion_batches(monkeypatch):
    snaps = [
        {
            "ts": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "bid_px": [1.0],
            "bid_qty": [1.0],
            "ask_px": [2.0],
            "ask_qty": [2.0],
        }
        for _ in range(5)
    ]

    inserted = []

    class DummyStorage:
        def insert_orderbook(self, engine, **data):
            inserted.append(data)

    storage = DummyStorage()
    worker = OrderBookBatchWorker(storage, engine="e", batch_size=2)

    flush_count = 0
    orig_flush = worker.flush

    async def counting_flush():
        nonlocal flush_count
        await orig_flush()
        flush_count += 1

    worker.flush = counting_flush

    adapter = DummyAdapter(snaps)
    await run_orderbook_ingestion(adapter, "BTC/USDT", depth=1, worker=worker)

    assert len(inserted) == 5
    assert flush_count == 3


@pytest.mark.asyncio
async def test_download_coinapi_open_interest_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyConnector:
        name = "dummy"

        async def fetch_open_interest(self, symbol, **params):
            return [{"timestamp": ts, "oi": 10.0}]

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_open_interest(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion._get_storage", lambda backend: DummyStorage()
    )
    await download_coinapi_open_interest(DummyConnector(), "BTCUSDT")
    assert inserted and inserted[0]["oi"] == 10.0


@pytest.mark.asyncio
async def test_download_kaiko_open_interest_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyConnector:
        name = "dummy"

        async def fetch_open_interest(self, exchange, pair, **params):
            return [{"timestamp": ts, "oi": 5.0}]

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_open_interest(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion._get_storage", lambda backend: DummyStorage()
    )
    await download_kaiko_open_interest(DummyConnector(), "ex", "BTCUSD")
    assert inserted and inserted[0]["oi"] == 5.0


@pytest.mark.asyncio
async def test_download_funding_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyConnector:
        name = "dummy"

        async def fetch_funding(self, symbol, **params):
            return {"ts": ts, "rate": 0.05, "interval_sec": 8}

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_funding(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion._get_storage", lambda backend: DummyStorage()
    )
    await download_funding(DummyConnector(), "BTCUSDT")
    assert inserted and inserted[0]["rate"] == 0.05


@pytest.mark.asyncio
async def test_download_kaiko_funding_persists(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyConnector:
        name = "dummy"

        async def fetch_funding(self, exchange, pair, **params):
            return [{"timestamp": ts, "rate": 0.02, "interval_sec": 12}]

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_funding(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(
        "tradingbot.data.ingestion._get_storage", lambda backend: DummyStorage()
    )
    await download_kaiko_funding(DummyConnector(), "ex", "BTCUSD")
    assert inserted and inserted[0]["rate"] == 0.02


@pytest.mark.asyncio
async def test_backfill_applies_rate_limit(monkeypatch):
    from tradingbot.jobs import backfill as backfill_job

    calls: list[str] = []

    class DummyExchange:
        rateLimit = 1200

        async def load_markets(self):
            self.symbols = ["BTC/USDT", "ETH/USDT"]

        async def fetch_ohlcv(self, symbol, timeframe, since, limit):
            calls.append(symbol)
            return []

        async def fetch_trades(self, symbol, since, limit):
            return []

        async def close(self):
            pass

    monkeypatch.setattr(
        backfill_job.ccxt, "binance", lambda params=None: DummyExchange()
    )

    class DummyClient:
        async def ensure_schema(self):
            pass

        def register_table(self, *args, **kwargs):
            pass

        async def add(self, *args, **kwargs):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(backfill_job, "AsyncTimescaleClient", DummyClient)

    sleeps: list[float] = []

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    monkeypatch.setattr(backfill_job.asyncio, "sleep", fake_sleep)

    await backfill_job.backfill(
        1,
        ["BTC/USDT", "ETH/USDT"],
        exchange_name="binance_spot",
        timeframe="3m",
    )

    assert calls == ["BTC/USDT", "ETH/USDT"]
    assert sleeps == []
