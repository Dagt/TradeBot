#!/usr/bin/env python3
"""Simple historical data downloader.

The script fetches 1-minute OHLCV bars using ``ccxt`` and persists them into
TimescaleDB or QuestDB depending on the selected backend.
"""

import asyncio
from datetime import datetime
from typing import Literal

import ccxt.async_support as ccxt
import typer

from tradingbot.storage import timescale as ts_storage
from tradingbot.storage import quest as qs_storage

Backend = Literal["timescale", "quest"]

app = typer.Typer()


def _get_storage(backend: Backend):
    return ts_storage if backend == "timescale" else qs_storage


async def _download_bars(exchange: str, symbol: str, start: datetime, end: datetime, backend: Backend):
    ex_class = getattr(ccxt, exchange)
    ex = ex_class()
    storage = _get_storage(backend)
    engine = storage.get_engine()
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while since < end_ms:
        ohlcvs = await ex.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=1000)
        if not ohlcvs:
            break
        for ts_ms, o, h, l, c, v in ohlcvs:
            ts = datetime.utcfromtimestamp(ts_ms / 1000)
            storage.insert_bar_1m(engine, exchange, symbol, ts, o, h, l, c, v)
        since = ohlcvs[-1][0] + 60_000
    await ex.close()


@app.command()
def bars(exchange: str, symbol: str, start: datetime, end: datetime, backend: Backend = "timescale"):
    """Download OHLCV bars into the selected backend."""
    asyncio.run(_download_bars(exchange, symbol, start, end, backend))


if __name__ == "__main__":
    app()
