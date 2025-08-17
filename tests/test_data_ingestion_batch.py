from datetime import datetime, timezone

import pytest

from tradingbot.workers import OrderBookBatchWorker, run_orderbook_ingestion


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
