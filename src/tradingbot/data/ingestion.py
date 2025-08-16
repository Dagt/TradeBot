import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from ..bus import EventBus
from ..types import Tick, Bar, OrderBook
from ..storage import timescale as ts_storage
from ..storage import quest as qs_storage

log = logging.getLogger(__name__)

Backends = Literal["timescale", "quest"]

def _get_storage(backend: Backends):
    """Return storage module depending on backend name."""
    return ts_storage if backend == "timescale" else qs_storage


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
