"""Backfill historical data and persist it using :class:`AsyncTimescaleClient`."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Sequence

import ccxt.async_support as ccxt
import logging

from ..storage.async_timescale import AsyncTimescaleClient

logger = logging.getLogger(__name__)


INSERT_BAR_SQL = """
    INSERT INTO market.bars
        (ts, timeframe, exchange, symbol, o, h, l, c, v)
    VALUES
        (:ts, :timeframe, :exchange, :symbol, :o, :h, :l, :c, :v)
    ON CONFLICT DO NOTHING
"""

INSERT_TRADE_SQL = """
    INSERT INTO market.trades
        (ts, exchange, symbol, px, qty, side, trade_id)
    VALUES
        (:ts, :exchange, :symbol, :px, :qty, :side, :trade_id)
    ON CONFLICT DO NOTHING
"""


async def _retry(func, *args, retries: int = 3, delay: float = 1.0, **kwargs):
    """Retry *func* with exponential backoff."""

    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception:  # pragma: no cover - network errors
            if attempt >= retries:
                raise
            await asyncio.sleep(delay * 2 ** (attempt - 1))


async def backfill(
    days: int,
    symbols: Sequence[str],
    exchange_name: str = "binance",
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    """Backfill OHLCV bars and trades for *symbols*.

    If *start* or *end* are not provided, the range defaults to the past
    ``days`` days ending at ``datetime.now(timezone.utc)``.
    """

    if end is None:
        end_dt = datetime.now(timezone.utc)
    else:
        end_dt = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end

    if start is None:
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start

    if start is not None and end is not None:
        logger.info(
            "Backfill start: %s \u2192 %s para %s",
            start_dt,
            end_dt,
            ", ".join(symbols),
        )
    else:
        logger.info("Backfill start: %d day(s) for %s", days, ", ".join(symbols))

    try:
        ex_class = getattr(ccxt, exchange_name)
    except AttributeError as exc:
        raise ValueError(f"Exchange {exchange_name!r} not supported") from exc

    ex = ex_class({"enableRateLimit": False})
    ex.id = exchange_name
    delay = getattr(ex, "rateLimit", 1000) / 1000

    client = AsyncTimescaleClient()
    await client.ensure_schema()
    client.register_table("market.bars", INSERT_BAR_SQL)
    client.register_table("market.trades", INSERT_TRADE_SQL)

    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)

    try:
        for symbol in symbols:
            logger.info("Procesando %s", symbol)
            db_symbol = symbol.replace("/", "")

            # --- OHLCV backfill -------------------------------------------------
            since = start_ms
            while since < end_ms:
                ohlcvs = await _retry(
                    ex.fetch_ohlcv,
                    symbol,
                    "1m",
                    since,
                    1000,
                    delay=delay,
                )
                await asyncio.sleep(delay)
                if not ohlcvs:
                    break

                for ts_ms, o, h, l, c, v in ohlcvs:
                    await client.add(
                        "market.bars",
                        {
                            "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                            "timeframe": "1m",
                            "exchange": ex.id,
                            "symbol": db_symbol,
                            "o": o,
                            "h": h,
                            "l": l,
                            "c": c,
                            "v": v,
                        },
                    )
                since = ohlcvs[-1][0] + 60_000

            # --- Trades backfill -----------------------------------------------
            since = start_ms
            while since < end_ms:
                trades = await _retry(
                    ex.fetch_trades, symbol, since, 1000, delay=delay
                )
                await asyncio.sleep(delay)
                if not trades:
                    break
                for t in trades:
                    await client.add(
                        "market.trades",
                        {
                            "ts": datetime.fromtimestamp(
                                t["timestamp"] / 1000, tz=timezone.utc
                            ),
                            "exchange": ex.id,
                            "symbol": db_symbol,
                            "px": t["price"],
                            "qty": t["amount"],
                            "side": t.get("side"),
                            "trade_id": t.get("id"),
                        },
                    )
                since = trades[-1]["timestamp"] + 1
            logger.info("Completado %s", symbol)

    finally:
        await client.stop()
        await ex.close()
        logger.info("Backfill finalizado para %s", ", ".join(symbols))

