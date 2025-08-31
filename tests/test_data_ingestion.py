import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.adapters.binance_futures import BinanceFuturesAdapter
from tradingbot.bus import EventBus
from tradingbot.data.ingestion import run_orderbook_stream, run_trades_stream
from tradingbot.data import ingestion
from tradingbot.types import OrderBook, Tick
from tradingbot.connectors import Trade as ConnTrade, OrderBook as ConnOrderBook
from tradingbot.core.symbols import normalize


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

    stream_order_book = stream_orderbook

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
        adapter, "BTC/USDT", depth=5, bus=bus, engine="engine", persist=True
    )

    assert len(published) == 1
    ob = published[0]
    assert isinstance(ob, OrderBook)
    assert ob.bid_px == [100.0, 99.5]
    assert len(inserted) == 1
    assert inserted[0]["symbol"] == normalize("BTC/USDT")
    assert inserted[0]["bid_px"] == [100.0, 99.5]


@pytest.mark.asyncio
async def test_run_orderbook_stream_no_persist(monkeypatch):
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

    called = False

    def _get_storage(backend):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(ingestion, "_get_storage", _get_storage)

    await run_orderbook_stream(
        adapter, "BTC/USDT", depth=5, bus=bus, engine=None, persist=False
    )

    assert len(published) == 1
    assert not called


@pytest.mark.asyncio
async def test_run_orderbook_stream_persists_bba(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    snapshot = {
        "ts": ts,
        "bid_px": [100.0],
        "bid_qty": [1.0],
        "ask_px": [100.5],
        "ask_qty": [1.5],
    }

    class DummyAdapter(ExchangeAdapter):
        name = "dummy"

        async def stream_trades(self, symbol: str):
            if False:
                yield {}

        async def stream_orderbook(self, symbol: str, depth: int):
            yield snapshot

        stream_order_book = stream_orderbook

        async def stream_bba(self, symbol: str):
            yield {
                "symbol": symbol,
                "ts": ts,
                "bid_px": snapshot["bid_px"][0],
                "bid_qty": snapshot["bid_qty"][0],
                "ask_px": snapshot["ask_px"][0],
                "ask_qty": snapshot["ask_qty"][0],
            }

        async def place_order(self, *args, **kwargs):
            return {}

        async def cancel_order(self, order_id: str):
            return {}

    inserted: list[dict] = []

    class DummyStorage:
        def insert_orderbook(self, engine, **data):
            pass

        def insert_bba(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    bus = EventBus()
    await run_orderbook_stream(
        DummyAdapter(),
        "BTC/USDT",
        depth=5,
        bus=bus,
        engine="engine",
        persist=True,
        emit_bba=True,
    )

    assert len(inserted) == 1
    rec = inserted[0]
    assert rec["bid_px"] == 100.0
    assert rec["bid_qty"] == 1.0
    assert rec["ask_px"] == 100.5
    assert rec["ask_qty"] == 1.5


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

        stream_order_book = stream_orderbook

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

        stream_order_book = stream_orderbook

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
@pytest.mark.parametrize("backend", ["timescale", "quest"])
async def test_poll_funding_inserts(monkeypatch, backend):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_funding(self, symbol: str):
            return {"ts": ts, "rate": 0.01, "interval_sec": 60}

    inserted = []
    requested_backends: list[str] = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_funding(self, engine, **data):
            inserted.append(data)

    def fake_get_storage(b):
        requested_backends.append(b)
        return DummyStorage()

    monkeypatch.setattr(ingestion, "_get_storage", fake_get_storage)

    task = asyncio.create_task(
        ingestion.poll_funding(DummyAdapter(), "BTC/USDT", interval=0, backend=backend)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inserted
    assert inserted[0]["rate"] == 0.01
    assert requested_backends == [backend]


@pytest.mark.asyncio
async def test_poll_open_interest_inserts(monkeypatch):
    ts_ms = 1_672_569_600_000  # 2023-01-01T00:00:00Z

    class DummyRest:
        def fapiPublicGetOpenInterest(self, params):
            return {"symbol": params.get("symbol"), "openInterest": "123.0", "time": ts_ms}

    adapter = BinanceFuturesAdapter.__new__(BinanceFuturesAdapter)
    adapter.rest = DummyRest()

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    inserted = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_open_interest(self, engine, **data):
            inserted.append(data)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    task = asyncio.create_task(
        ingestion.poll_open_interest(adapter, "BTC/USDT", interval=0)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inserted
    assert inserted[0]["oi"] == 123.0
    assert inserted[0]["ts"] == datetime.fromtimestamp(ts_ms / 1000, timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["timescale", "quest"])
async def test_poll_basis_inserts(monkeypatch, backend):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyAdapter:
        name = "dummy"

        async def fetch_basis(self, symbol: str):
            return {"ts": ts, "basis": 5.0}

    inserted = []
    requested_backends: list[str] = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_basis(self, engine, **data):
            inserted.append(data)

    def fake_get_storage(b):
        requested_backends.append(b)
        return DummyStorage()

    monkeypatch.setattr(ingestion, "_get_storage", fake_get_storage)

    task = asyncio.create_task(
        ingestion.poll_basis(DummyAdapter(), "BTC/USDT", interval=0, backend=backend)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inserted
    assert inserted[0]["basis"] == 5.0
    assert inserted[0]["ts"] == ts
    assert requested_backends == [backend]


@pytest.mark.asyncio
async def test_download_trades_connector(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyConn:
        name = "kaiko"

        async def fetch_trades(self, symbol: str, **params):
            return [
                ConnTrade(
                    timestamp=ts,
                    exchange="kaiko",
                    symbol=symbol,
                    price=1.0,
                    amount=2.0,
                    side="buy",
                )
            ]

    captured: list[Tick] = []

    def fake_persist(trades, *, backend, path=None):
        captured.extend(trades)

    monkeypatch.setattr(ingestion, "persist_trades", fake_persist)

    await ingestion.download_trades(DummyConn(), "BTC/USDT", backend="csv")

    assert captured
    assert captured[0].price == 1.0
    assert captured[0].qty == 2.0


@pytest.mark.asyncio
async def test_download_order_book_connector(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    class DummyConn:
        name = "kaiko"

        async def fetch_order_book(self, symbol: str, **params):
            return ConnOrderBook(
                timestamp=ts,
                exchange="kaiko",
                symbol=symbol,
                bids=[(100.0, 1.0)],
                asks=[(101.0, 2.0)],
            )

    captured: list[OrderBook] = []

    def fake_persist(obs, *, backend, path=None):
        captured.extend(obs)

    monkeypatch.setattr(ingestion, "persist_orderbooks", fake_persist)

    await ingestion.download_order_book(DummyConn(), "BTC/USDT", backend="csv")

    assert captured
    assert captured[0].bid_px == [100.0]
    assert captured[0].ask_qty == [2.0]


@pytest.mark.asyncio
async def test_fetch_bars_no_persist(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    ts_ms = int(ts.timestamp() * 1000)

    class DummyAdapter:
        name = "dummy"

        async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
            return [(ts_ms, 1.0, 2.0, 0.5, 1.5, 100.0)]

    called = False
    inserted: list[dict] = []

    def fake_get_storage(backend):
        nonlocal called
        called = True

        class DummyStorage:
            def get_engine(self):
                return "engine"

            def insert_bar(self, *args, **kwargs):
                inserted.append(kwargs)

        return DummyStorage()

    monkeypatch.setattr(ingestion, "_get_storage", fake_get_storage)

    task = asyncio.create_task(
        ingestion.fetch_bars(
            DummyAdapter(),
            "BTC/USDT",
            persist=False,
            sleep_s=999,
            log=False,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not called
    assert inserted == []


@pytest.mark.asyncio
async def test_fetch_bars_persist_inserts(monkeypatch):
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    ts_ms = int(ts.timestamp() * 1000)

    class DummyAdapter:
        name = "dummy"

        async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
            return [(ts_ms, 1.0, 2.0, 0.5, 1.5, 100.0)]

    inserted: list[tuple] = []

    class DummyStorage:
        def get_engine(self):
            return "engine"

        def insert_bar(self, *args):
            inserted.append(args)

    monkeypatch.setattr(ingestion, "_get_storage", lambda backend: DummyStorage())

    task = asyncio.create_task(
        ingestion.fetch_bars(
            DummyAdapter(),
            "BTC/USDT",
            persist=True,
            sleep_s=999,
            log=False,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(inserted) == 1

def test_cli_ingest_csv_creates_file(monkeypatch, tmp_path):
    """Running the ingest CLI with CSV backend should create a CSV file."""

    monkeypatch.setattr(
        "tradingbot.utils.time_sync.check_ntp_offset", lambda: 0.0
    )
    import importlib
    from typer.testing import CliRunner
    import tradingbot.cli.main as cli_main

    cli_main = importlib.reload(cli_main)

    def fail():
        raise AssertionError("get_engine should not be called for CSV backend")
    monkeypatch.setattr("tradingbot.storage.timescale.get_engine", fail)
    monkeypatch.setattr("tradingbot.storage.quest.get_engine", fail)

    class DummyAdapter:
        name = "dummy"

        async def stream_order_book(self, symbol: str, depth: int):
            yield {
                "ts": datetime(2023, 1, 1, tzinfo=timezone.utc),
                "bid_px": [1.0],
                "bid_qty": [2.0],
                "ask_px": [3.0],
                "ask_qty": [4.0],
            }

    monkeypatch.setattr(cli_main, "get_adapter_class", lambda name: DummyAdapter)
    monkeypatch.setattr(cli_main, "_AVAILABLE_VENUES", {"dummy"})

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        cli_main.app,
        [
            "ingest",
            "--venue",
            "dummy",
            "--symbol",
            "BTC/USDT",
            "--kind",
            "orderbook",
            "--backend",
            "csv",
            "--persist",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "db" / "orderbook.csv").exists()
