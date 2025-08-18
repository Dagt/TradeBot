#!/usr/bin/env python3
"""Simple historical data downloader.

The script fetches 1-minute OHLCV bars, trades and level 2 order book snapshots
using ``ccxt`` and persists them into TimescaleDB, QuestDB or CSV files in
``db/``.
"""

import asyncio
from datetime import datetime
from typing import Literal

import ccxt.async_support as ccxt
import typer

from tradingbot.data import ingestion
from tradingbot.types import Bar, OrderBook, Tick

Backend = Literal["timescale", "quest", "csv"]

app = typer.Typer()


async def _download_bars(
    exchange: str, symbol: str, start: datetime, end: datetime, backend: Backend
) -> None:
    ex_class = getattr(ccxt, exchange)
    ex = ex_class()
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while since < end_ms:
        ohlcvs = await ex.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=1000)
        if not ohlcvs:
            break
        bars: list[Bar] = []
        for ts_ms, o, h, l, c, v in ohlcvs:
            ts = datetime.utcfromtimestamp(ts_ms / 1000)
            bars.append(
                Bar(
                    ts=ts,
                    timeframe="1m",
                    exchange=exchange,
                    symbol=symbol,
                    o=o,
                    h=h,
                    l=l,
                    c=c,
                    v=v,
                )
            )
        ingestion.persist_bars(bars, backend=backend)
        since = ohlcvs[-1][0] + 60_000
    await ex.close()


async def _download_trades(
    exchange: str, symbol: str, start: datetime, end: datetime, backend: Backend
) -> None:
    ex_class = getattr(ccxt, exchange)
    ex = ex_class()
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while since < end_ms:
        data = await ex.fetch_trades(symbol, since=since, limit=1000)
        if not data:
            break
        ticks: list[Tick] = []
        for t in data:
            ticks.append(
                Tick(
                    ts=datetime.utcfromtimestamp(t["timestamp"] / 1000),
                    exchange=exchange,
                    symbol=symbol,
                    price=float(t["price"]),
                    qty=float(t["amount"]),
                    side=t.get("side"),
                )
            )
        ingestion.persist_trades(ticks, backend=backend)
        since = data[-1]["timestamp"] + 1
    await ex.close()


async def _download_l2(
    exchange: str,
    symbol: str,
    snapshots: int,
    depth: int,
    interval: float,
    backend: Backend,
) -> None:
    ex_class = getattr(ccxt, exchange)
    ex = ex_class()
    for _ in range(snapshots):
        ob = await ex.fetch_order_book(symbol, limit=depth)
        bids = ob.get("bids", [])[:depth]
        asks = ob.get("asks", [])[:depth]
        snapshot = OrderBook(
            ts=datetime.utcnow(),
            exchange=exchange,
            symbol=symbol,
            bid_px=[float(p) for p, _ in bids],
            bid_qty=[float(q) for _, q in bids],
            ask_px=[float(p) for p, _ in asks],
            ask_qty=[float(q) for _, q in asks],
        )
        ingestion.persist_orderbooks([snapshot], backend=backend)
        await asyncio.sleep(interval)
    await ex.close()


@app.command()
def bars(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    backend: Backend = "timescale",
) -> None:
    """Download OHLCV bars into the selected backend."""
    asyncio.run(_download_bars(exchange, symbol, start, end, backend))


@app.command()
def trades(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    backend: Backend = "timescale",
) -> None:
    """Download recent trades into storage."""
    asyncio.run(_download_trades(exchange, symbol, start, end, backend))


@app.command()
def l2(
    exchange: str,
    symbol: str,
    snapshots: int = 10,
    depth: int = 10,
    interval: float = 1.0,
    backend: Backend = "timescale",
) -> None:
    """Collect level 2 order book snapshots."""
    asyncio.run(_download_l2(exchange, symbol, snapshots, depth, interval, backend))


if __name__ == "__main__":
    app()
