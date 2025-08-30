#!/usr/bin/env python3
"""CLI utilities for backfilling market data.

Commands in this module use ``tradingbot.connectors`` (wrapping ``ccxt``)
for data retrieval and persist the results into TimescaleDB or QuestDB.
Basic retry and rate limit handling are provided via the connectors and
``tradingbot.utils.retry.with_retry``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Literal

import ccxt.async_support as ccxt
import typer

from tradingbot.connectors.binance import BinanceConnector
from tradingbot.connectors.bybit import BybitConnector
from tradingbot.connectors.okx import OkxConnector
from tradingbot.connectors.base import Trade
from tradingbot.storage import quest as qs_storage
from tradingbot.storage import timescale as ts_storage
from tradingbot.utils.retry import with_retry

logging.basicConfig(level=logging.INFO)

Backend = Literal["timescale", "quest"]
app = typer.Typer(help="Backfill market data into TimescaleDB or QuestDB")

CONNECTORS: dict[str, type] = {
    "binance": BinanceConnector,
    "bybit": BybitConnector,
    "okx": OkxConnector,
}


def _get_storage(backend: Backend):
    return ts_storage if backend == "timescale" else qs_storage


async def _download_trades(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    backend: Backend,
):
    log = logging.getLogger("trades")
    conn_cls = CONNECTORS[exchange]
    conn = conn_cls()
    storage = _get_storage(backend)
    engine = storage.get_engine()

    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while since < end_ms:
        data = await with_retry(conn.rest.fetch_trades, symbol, since=since, limit=1000)
        if not data:
            break
        for t in data:
            tr = Trade(
                timestamp=datetime.utcfromtimestamp(t["timestamp"] / 1000),
                exchange=conn.name,
                symbol=symbol,
                price=float(t["price"]),
                amount=float(t["amount"]),
                side=t.get("side", ""),
            )
            storage.insert_trade(engine, tr)
        log.info("inserted %d trades up to %s", len(data), tr.timestamp)
        since = data[-1]["timestamp"] + 1
    await conn.rest.close()


async def _download_bars(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    backend: Backend,
):
    log = logging.getLogger("ohlcv")
    ex_class = getattr(ccxt, exchange)
    ex = ex_class({"enableRateLimit": True})
    storage = _get_storage(backend)
    engine = storage.get_engine()

    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while since < end_ms:
        ohlcvs = await with_retry(
            ex.fetch_ohlcv, symbol, timeframe="3m", since=since, limit=1000
        )
        if not ohlcvs:
            break
        for ts_ms, o, h, l, c, v in ohlcvs:
            ts = datetime.utcfromtimestamp(ts_ms / 1000)
            storage.insert_bar(engine, exchange, symbol, ts, "3m", o, h, l, c, v)
        log.info("inserted %d bars up to %s", len(ohlcvs), ts)
        since = ohlcvs[-1][0] + 180_000
    await ex.close()


async def _download_l2(
    exchange: str,
    symbol: str,
    snapshots: int,
    depth: int,
    interval: float,
    backend: Backend,
):
    log = logging.getLogger("l2")
    conn_cls = CONNECTORS[exchange]
    conn = conn_cls()
    storage = _get_storage(backend)
    engine = storage.get_engine()

    for i in range(snapshots):
        ob = await with_retry(conn.rest.fetch_order_book, symbol, limit=depth)
        ts = datetime.utcnow()
        bids = ob.get("bids", [])[:depth]
        asks = ob.get("asks", [])[:depth]
        storage.insert_orderbook(
            engine,
            ts=ts,
            exchange=exchange,
            symbol=symbol,
            bid_px=[float(p) for p, _ in bids],
            bid_qty=[float(q) for _, q in bids],
            ask_px=[float(p) for p, _ in asks],
            ask_qty=[float(q) for _, q in asks],
        )
        log.info("snapshot %d stored at %s", i + 1, ts)
        await asyncio.sleep(interval)
    await conn.rest.close()


@app.command()
def trades(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    backend: Backend = "timescale",
):
    """Backfill recent trades into storage."""
    asyncio.run(_download_trades(exchange, symbol, start, end, backend))


@app.command()
def ohlcv(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    backend: Backend = "timescale",
):
    """Backfill 3m OHLCV bars."""
    asyncio.run(_download_bars(exchange, symbol, start, end, backend))


@app.command()
def l2(
    exchange: str,
    symbol: str,
    snapshots: int = 10,
    depth: int = 10,
    interval: float = 1.0,
    backend: Backend = "timescale",
):
    """Collect Level 2 order book snapshots."""
    asyncio.run(_download_l2(exchange, symbol, snapshots, depth, interval, backend))


if __name__ == "__main__":
    app()
