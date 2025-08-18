from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Sequence

import ccxt.async_support as ccxt


async def backfill(days: int, symbols: Sequence[str]) -> None:
    """Backfill minute bars for ``symbols`` over the past ``days`` days.

    A simple rate limit is enforced between requests based on the exchange's
    ``rateLimit`` attribute.  Data returned by the exchange is discarded; this
    helper merely demonstrates sequential API calls with throttling.
    """

    ex = ccxt.binance({"enableRateLimit": False})
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(timedelta(days=days).total_seconds() * 1000)

    for symbol in symbols:
        since = start_ms
        while since < end_ms:
            ohlcvs = await ex.fetch_ohlcv(
                symbol, timeframe="1m", since=since, limit=1000
            )
            await asyncio.sleep(getattr(ex, "rateLimit", 1000) / 1000)
            if not ohlcvs:
                break
            since = ohlcvs[-1][0] + 60_000
    await ex.close()

