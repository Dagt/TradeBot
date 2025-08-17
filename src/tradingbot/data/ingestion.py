import asyncio
import csv
from pathlib import Path
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from ..bus import EventBus
from ..types import Tick, Bar, OrderBook
from ..connectors import Trade as ConnTrade, OrderBook as ConnOrderBook
from ..connectors.kaiko import KaikoConnector
from ..connectors.coinapi import CoinAPIConnector
from ..storage import timescale as ts_storage
from ..storage import quest as qs_storage

log = logging.getLogger(__name__)

Backends = Literal["timescale", "quest", "csv"]


def _get_storage(backend: Backends):
    """Return storage module depending on backend name."""
    if backend == "csv":
        return None
    return ts_storage if backend == "timescale" else qs_storage


def _csv_path(name: str, path: Path | None = None) -> Path:
    """Return a path inside ``db/`` for CSV persistence."""
    if path is None:
        path = Path("db") / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def persist_trades(trades: Iterable[Tick], *, backend: Backends = "timescale", path: Path | None = None) -> None:
    """Persist an iterable of trades to the selected backend or CSV file."""
    if backend == "csv":
        csv_path = _csv_path("trades", path)
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for t in trades:
                writer.writerow([t.ts.isoformat(), t.exchange, t.symbol, t.price, t.qty, t.side or ""])
        return

    storage = _get_storage(backend)
    if storage is None:
        return
    engine = storage.get_engine()
    for t in trades:
        try:
            storage.insert_trade(engine, t)
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Trade insert failed: %s", exc)


def persist_orderbooks(orderbooks: Iterable[OrderBook], *, backend: Backends = "timescale", path: Path | None = None) -> None:
    """Persist order book snapshots to the selected backend or CSV file."""
    if backend == "csv":
        csv_path = _csv_path("orderbook", path)
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for ob in orderbooks:
                writer.writerow(
                    [
                        ob.ts.isoformat(),
                        ob.exchange,
                        ob.symbol,
                        ";".join(map(str, ob.bid_px)),
                        ";".join(map(str, ob.bid_qty)),
                        ";".join(map(str, ob.ask_px)),
                        ";".join(map(str, ob.ask_qty)),
                    ]
                )
        return

    storage = _get_storage(backend)
    if storage is None:
        return
    engine = storage.get_engine()
    for ob in orderbooks:
        try:
            storage.insert_orderbook(
                engine,
                ts=ob.ts,
                exchange=ob.exchange,
                symbol=ob.symbol,
                bid_px=ob.bid_px,
                bid_qty=ob.bid_qty,
                ask_px=ob.ask_px,
                ask_qty=ob.ask_qty,
            )
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Orderbook insert failed: %s", exc)


def persist_bars(bars: Iterable[Bar], *, backend: Backends = "timescale", path: Path | None = None) -> None:
    """Persist OHLCV bars to the selected backend or CSV file."""
    if backend == "csv":
        csv_path = _csv_path("bars", path)
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for b in bars:
                writer.writerow([b.ts.isoformat(), b.exchange, b.symbol, b.o, b.h, b.l, b.c, b.v])
        return

    storage = _get_storage(backend)
    if storage is None:
        return
    engine = storage.get_engine()
    for b in bars:
        try:
            storage.insert_bar_1m(engine, b.exchange, b.symbol, b.ts, b.o, b.h, b.l, b.c, b.v)
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Bar insert failed: %s", exc)


def persist_funding(
    fundings: Iterable[dict[str, Any]], *, backend: Backends = "timescale", path: Path | None = None
) -> None:
    """Persist funding rates to the selected backend or CSV file."""
    if backend == "csv":
        csv_path = _csv_path("funding", path)
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for f in fundings:
                writer.writerow(
                    [
                        f["ts"].isoformat(),
                        f["exchange"],
                        f["symbol"],
                        f["rate"],
                        f.get("interval_sec", 0),
                    ]
                )
        return

    storage = _get_storage(backend)
    if storage is None:
        return
    engine = storage.get_engine()
    for f in fundings:
        try:
            storage.insert_funding(
                engine,
                ts=f["ts"],
                exchange=f["exchange"],
                symbol=f["symbol"],
                rate=f["rate"],
                interval_sec=f.get("interval_sec", 0),
            )
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Funding insert failed: %s", exc)


def persist_open_interest(
    records: Iterable[dict[str, Any]], *, backend: Backends = "timescale", path: Path | None = None
) -> None:
    """Persist open interest snapshots to the selected backend or CSV file."""
    if backend == "csv":
        csv_path = _csv_path("open_interest", path)
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for r in records:
                writer.writerow([r["ts"].isoformat(), r["exchange"], r["symbol"], r["oi"]])
        return

    storage = _get_storage(backend)
    if storage is None:
        return
    engine = storage.get_engine()
    for r in records:
        try:
            storage.insert_open_interest(
                engine,
                ts=r["ts"],
                exchange=r["exchange"],
                symbol=r["symbol"],
                oi=r["oi"],
            )
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Open interest insert failed: %s", exc)


async def download_trades(
    connector: Any,
    symbol: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch trades using *connector* and persist them.

    The connector must implement a ``fetch_trades`` coroutine returning a list of
    :class:`tradingbot.connectors.Trade` objects.
    """

    raw_trades: Iterable[ConnTrade] = await connector.fetch_trades(symbol, **params)
    ticks: list[Tick] = []
    for t in raw_trades:
        ticks.append(
            Tick(
                ts=t.timestamp,
                exchange=getattr(connector, "name", "unknown"),
                symbol=symbol,
                price=t.price,
                qty=t.amount,
                side=t.side,
            )
        )
    persist_trades(ticks, backend=backend)


async def download_order_book(
    connector: Any,
    symbol: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch an order book snapshot using *connector* and persist it."""

    ob: ConnOrderBook = await connector.fetch_order_book(symbol, **params)
    snapshot = OrderBook(
        ts=ob.timestamp,
        exchange=getattr(connector, "name", "unknown"),
        symbol=symbol,
        bid_px=[p for p, _ in ob.bids],
        bid_qty=[q for _, q in ob.bids],
        ask_px=[p for p, _ in ob.asks],
        ask_qty=[q for _, q in ob.asks],
    )
    persist_orderbooks([snapshot], backend=backend)


async def download_funding(
    connector: Any,
    symbol: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch a funding rate using *connector* and persist it."""

    info: Any = await connector.fetch_funding(symbol, **params)
    ts = getattr(info, "timestamp", None) or getattr(info, "ts", None) or info.get("ts")
    rate = getattr(info, "rate", None) or info.get("rate") or info.get("fundingRate")
    interval = getattr(info, "interval_sec", None) or info.get("interval_sec") or info.get("interval") or 0
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, timezone.utc)
    record = {
        "ts": ts,
        "exchange": getattr(connector, "name", "unknown"),
        "symbol": symbol,
        "rate": float(rate or 0.0),
        "interval_sec": int(interval),
    }
    persist_funding([record], backend=backend)


async def download_kaiko_trades(
    connector: KaikoConnector,
    exchange: str,
    pair: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch trades from Kaiko and persist them."""

    raw_trades: Iterable[ConnTrade] = await connector.fetch_trades(
        exchange, pair, **params
    )
    ticks = [
        Tick(
            ts=t.timestamp,
            exchange=exchange,
            symbol=pair,
            price=t.price,
            qty=t.amount,
            side=t.side,
        )
        for t in raw_trades
    ]
    persist_trades(ticks, backend=backend)


async def download_kaiko_order_book(
    connector: KaikoConnector,
    exchange: str,
    pair: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch an order book snapshot from Kaiko and persist it."""

    ob: ConnOrderBook = await connector.fetch_order_book(exchange, pair, **params)
    snapshot = OrderBook(
        ts=ob.timestamp,
        exchange=exchange,
        symbol=pair,
        bid_px=[p for p, _ in ob.bids],
        bid_qty=[q for _, q in ob.bids],
        ask_px=[p for p, _ in ob.asks],
        ask_qty=[q for _, q in ob.asks],
    )
    persist_orderbooks([snapshot], backend=backend)


async def download_kaiko_open_interest(
    connector: KaikoConnector,
    exchange: str,
    pair: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch open interest from Kaiko and persist it."""

    raw = await connector.fetch_open_interest(exchange, pair, **params)
    records = []
    for r in raw if isinstance(raw, Iterable) else [raw]:
        ts = getattr(r, "timestamp", None) or getattr(r, "ts", None) or r.get("ts")
        oi = getattr(r, "oi", None) or r.get("oi") or r.get("openInterest")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, timezone.utc)
        records.append({"ts": ts, "exchange": exchange, "symbol": pair, "oi": float(oi or 0.0)})
    persist_open_interest(records, backend=backend)


async def download_coinapi_trades(
    connector: CoinAPIConnector,
    symbol: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch trades from CoinAPI and persist them."""

    raw_trades: Iterable[ConnTrade] = await connector.fetch_trades(symbol, **params)
    ticks = [
        Tick(
            ts=t.timestamp,
            exchange=connector.name,
            symbol=symbol,
            price=t.price,
            qty=t.amount,
            side=t.side,
        )
        for t in raw_trades
    ]
    persist_trades(ticks, backend=backend)


async def download_coinapi_order_book(
    connector: CoinAPIConnector,
    symbol: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch an order book snapshot from CoinAPI and persist it."""

    ob: ConnOrderBook = await connector.fetch_order_book(symbol, **params)
    snapshot = OrderBook(
        ts=ob.timestamp,
        exchange=connector.name,
        symbol=symbol,
        bid_px=[p for p, _ in ob.bids],
        bid_qty=[q for _, q in ob.bids],
        ask_px=[p for p, _ in ob.asks],
        ask_qty=[q for _, q in ob.asks],
    )
    persist_orderbooks([snapshot], backend=backend)


async def download_coinapi_open_interest(
    connector: CoinAPIConnector,
    symbol: str,
    *,
    backend: Backends = "timescale",
    **params: Any,
) -> None:
    """Fetch open interest from CoinAPI and persist it."""

    raw = await connector.fetch_open_interest(symbol, **params)
    records = []
    for r in raw if isinstance(raw, Iterable) else [raw]:
        ts = getattr(r, "timestamp", None) or getattr(r, "ts", None) or r.get("ts")
        oi = getattr(r, "oi", None) or r.get("oi") or r.get("openInterest")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, timezone.utc)
        records.append({"ts": ts, "exchange": connector.name, "symbol": symbol, "oi": float(oi or 0.0)})
    persist_open_interest(records, backend=backend)


async def run_trades_stream(adapter: Any, symbol: str, bus: EventBus) -> None:
    """Publish trades from *adapter* to an :class:`EventBus`."""
    async for d in adapter.stream_trades(symbol):
        tick = Tick(
            ts=d.get("ts", datetime.now(timezone.utc)),
            exchange=getattr(adapter, "name", "unknown"),
            symbol=symbol,
            price=float(d.get("price") or d.get("px") or 0.0),
            qty=float(d.get("qty") or d.get("size") or 0.0),
            side=d.get("side"),
        )
        await bus.publish("trades", tick)


async def run_orderbook_stream(
    adapter: Any,
    symbol: str,
    depth: int,
    bus: EventBus,
    engine: Any,
    *,
    backend: Backends = "timescale",
) -> None:
    """Publish and persist order book snapshots."""
    storage = _get_storage(backend)
    async for d in adapter.stream_order_book(symbol, depth):
        ob = OrderBook(
            ts=d.get("ts", datetime.now(timezone.utc)),
            exchange=getattr(adapter, "name", "unknown"),
            symbol=symbol,
            bid_px=d.get("bid_px") or [],
            bid_qty=d.get("bid_qty") or [],
            ask_px=d.get("ask_px") or [],
            ask_qty=d.get("ask_qty") or [],
        )
        await bus.publish("orderbook", ob)
        storage.insert_orderbook(
            engine,
            ts=ob.ts,
            exchange=ob.exchange,
            symbol=ob.symbol,
            bid_px=ob.bid_px,
            bid_qty=ob.bid_qty,
            ask_px=ob.ask_px,
            ask_qty=ob.ask_qty,
        )

async def stream_trades(adapter: Any, symbol: str, *, backend: Backends = "timescale"):
    """Stream trades from *adapter* and persist each tick.

    Parameters
    ----------
    adapter: object with ``stream_trades`` async generator and ``name`` attribute.
    symbol: market symbol to subscribe.
    backend: choose ``"timescale"`` or ``"quest"`` storage backend.
    """
    storage = _get_storage(backend)
    engine = storage.get_engine()
    async for d in adapter.stream_trades(symbol):
        tick = Tick(
            ts=d.get("ts", datetime.now(timezone.utc)),
            exchange=getattr(adapter, "name", "unknown"),
            symbol=symbol,
            price=float(d.get("price") or d.get("px") or 0.0),
            qty=float(d.get("qty") or d.get("size") or 0.0),
            side=d.get("side"),
        )
        try:
            storage.insert_trade(engine, tick)
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Trade insert failed: %s", exc)

async def stream_orderbook(adapter: Any, symbol: str, depth: int = 10, *, backend: Backends = "timescale"):
    """Stream L2 orderbook snapshots and persist them."""
    storage = _get_storage(backend)
    engine = storage.get_engine()
    async for d in adapter.stream_order_book(symbol, depth):
        data = {
            "ts": d.get("ts", datetime.now(timezone.utc)),
            "exchange": getattr(adapter, "name", "unknown"),
            "symbol": symbol,
            "bid_px": d.get("bid_px") or [],
            "bid_qty": d.get("bid_qty") or [],
            "ask_px": d.get("ask_px") or [],
            "ask_qty": d.get("ask_qty") or [],
        }
        try:
            storage.insert_orderbook(engine, **data)
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Orderbook insert failed: %s", exc)

async def poll_funding(adapter: Any, symbol: str, *, interval: int = 60, backend: Backends = "timescale"):
    """Poll periodic funding rates and persist them."""
    storage = _get_storage(backend)
    engine = storage.get_engine()
    while True:
        try:
            info: Any = await adapter.fetch_funding(symbol)
            ts = info.get("ts", datetime.now(timezone.utc))
            rate = float(info.get("rate") or info.get("fundingRate") or 0.0)
            interval_sec = int(info.get("interval_sec") or info.get("interval", 0))
            storage.insert_funding(
                engine,
                ts=ts,
                exchange=getattr(adapter, "name", "unknown"),
                symbol=symbol,
                rate=rate,
                interval_sec=interval_sec,
            )
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Funding poll failed: %s", exc)
        await asyncio.sleep(interval)


async def poll_open_interest(
    adapter: Any,
    symbol: str,
    *,
    interval: int = 60,
    backend: Backends = "timescale",
):
    """Poll periodic open interest and persist it."""
    storage = _get_storage(backend)
    engine = storage.get_engine()
    while True:
        try:
            info: Any = await adapter.fetch_oi(symbol)
            ts = info.get("ts", datetime.now(timezone.utc))
            oi = float(info.get("oi") or info.get("openInterest") or 0.0)
            storage.insert_open_interest(
                engine,
                ts=ts,
                exchange=getattr(adapter, "name", "unknown"),
                symbol=symbol,
                oi=oi,
            )
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Open interest poll failed: %s", exc)
        await asyncio.sleep(interval)


async def poll_basis(
    adapter: Any,
    symbol: str,
    *,
    interval: int = 60,
    backend: Backends = "timescale",
):
    """Poll periodic basis and persist it."""
    storage = _get_storage(backend)
    engine = storage.get_engine()
    while True:
        try:
            info: Any = await adapter.fetch_basis(symbol)
            ts = info.get("ts", datetime.now(timezone.utc))
            basis = float(info.get("basis") or 0.0)
            storage.insert_basis(
                engine,
                ts=ts,
                exchange=getattr(adapter, "name", "unknown"),
                symbol=symbol,
                basis=basis,
            )
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Basis poll failed: %s", exc)
        await asyncio.sleep(interval)

async def fetch_bars(adapter: Any, symbol: str, *, timeframe: str = "1m", backend: Backends = "timescale", sleep_s: int = 60):
    """Periodically fetch OHLCV bars and persist them."""
    storage = _get_storage(backend)
    engine = storage.get_engine()
    while True:
        try:
            bars = await adapter.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            for ts_ms, o, h, l, c, v in bars:
                ts = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
                bar = Bar(ts=ts, timeframe=timeframe, exchange=getattr(adapter, "name", "unknown"), symbol=symbol, o=o, h=h, l=l, c=c, v=v)
                storage.insert_bar_1m(engine, bar.exchange, bar.symbol, bar.ts, bar.o, bar.h, bar.l, bar.c, bar.v)
        except Exception as exc:  # pragma: no cover - logging only
            log.debug("Bar fetch failed: %s", exc)
        await asyncio.sleep(sleep_s)


async def poll_perp_metrics(
    adapter: Any,
    symbol: str,
    *,
    interval: int = 60,
    backend: Backends = "timescale",
) -> None:
    """Run perpetual data pollers concurrently.

    This helper gathers :func:`poll_funding`, :func:`poll_basis` and
    :func:`poll_open_interest` so that funding rates, basis and open interest
    are fetched continuously.
    """

    await asyncio.gather(
        poll_funding(adapter, symbol, interval=interval, backend=backend),
        poll_basis(adapter, symbol, interval=interval, backend=backend),
        poll_open_interest(adapter, symbol, interval=interval, backend=backend),
    )
