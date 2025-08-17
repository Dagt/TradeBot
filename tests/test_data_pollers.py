import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.bus import EventBus
from tradingbot.data.open_interest import poll_open_interest
from tradingbot.data.funding import poll_funding
from tradingbot.data.basis import poll_basis
from tradingbot.connectors import Funding, OpenInterest
from tradingbot.utils.metrics import BASIS, BASIS_HIST


@pytest.mark.asyncio
async def test_poll_open_interest_publishes_and_inserts(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_open_interest(self, symbol: str):
            return OpenInterest(
                timestamp=ts, exchange="dummy", symbol=symbol, oi=10.0
            )

    events = []
    bus = EventBus()
    bus.subscribe("open_interest", lambda e: events.append(e))

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_open_interest(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr("tradingbot.data.open_interest.get_engine", lambda: "engine")
    monkeypatch.setattr("tradingbot.data.open_interest.insert_open_interest", lambda *a, **k: inserted.append(k))
    monkeypatch.setattr("tradingbot.data.open_interest._CAN_PG", True)

    task = asyncio.create_task(
        poll_open_interest(DummyAdapter(), "BTCUSDT", bus, interval=0, persist_pg=True)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events and events[0]["oi"] == 10.0
    assert inserted and inserted[0]["oi"] == 10.0


@pytest.mark.asyncio
async def test_poll_funding_publishes_and_inserts(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_funding(self, symbol: str):
            return Funding(
                timestamp=ts, exchange="dummy", symbol=symbol, rate=0.01
            )

    events = []
    bus = EventBus()
    bus.subscribe("funding", lambda e: events.append(e))

    inserted = []

    monkeypatch.setattr("tradingbot.data.funding.get_engine", lambda: "engine")
    monkeypatch.setattr(
        "tradingbot.data.funding.insert_funding",
        lambda *a, **k: inserted.append(k),
    )
    monkeypatch.setattr("tradingbot.data.funding._CAN_PG", True)

    task = asyncio.create_task(
        poll_funding(DummyAdapter(), "BTCUSDT", bus, interval=0, persist_pg=True)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events and events[0]["rate"] == 0.01
    assert inserted and inserted[0]["rate"] == 0.01


@pytest.mark.asyncio
async def test_poll_basis_publishes_and_inserts(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_basis(self, symbol: str):
            return {"ts": ts, "basis": 5.0}

    events = []
    inserted = []
    bus = EventBus()
    bus.subscribe("basis", lambda e: events.append(e))

    monkeypatch.setattr("tradingbot.data.basis.get_engine", lambda: "engine")
    monkeypatch.setattr("tradingbot.data.basis.insert_basis", lambda *a, **k: inserted.append(k))
    monkeypatch.setattr("tradingbot.data.basis._CAN_PG", True)

    BASIS.clear()
    BASIS_HIST.clear()

    task = asyncio.create_task(
        poll_basis(DummyAdapter(), "BTCUSDT", bus, interval=1, persist_pg=True)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events and events[0]["basis"] == 5.0
    assert inserted and inserted[0]["basis"] == 5.0
    assert BASIS.labels(symbol="BTCUSDT")._value.get() == 5.0
    hist_samples = [
        s
        for m in BASIS_HIST.collect()
        for s in m.samples
        if s.name.endswith("_sum")
    ]
    assert hist_samples and hist_samples[0].value == 5.0
