from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from tradingbot.storage import AsyncDBClient

SCHEMA_SQL = Path("db/schema.sql").read_text()


@pytest_asyncio.fixture(scope="module")
async def engine():
    dsn = "postgresql+asyncpg://postgres:postgres@localhost/tradebot_test"
    eng = create_async_engine(dsn, echo=False)
    async with eng.begin() as conn:
        for stmt in SCHEMA_SQL.split(";"):
            s = stmt.strip()
            if s:
                await conn.execute(text(s))
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def client(engine):
    cli = AsyncDBClient(dsn="postgresql+asyncpg://postgres:postgres@localhost/tradebot_test")
    await cli.connect()
    yield cli
    await cli.close()


@pytest.mark.asyncio
async def test_insert_and_query_trades(client):
    worker_sql = """
        INSERT INTO market.trades (ts, exchange, symbol, px, qty, side, trade_id)
        VALUES (:ts, :exchange, :symbol, :px, :qty, :side, :trade_id)
    """
    from tradingbot.workers import BatchIngestionWorker

    worker = BatchIngestionWorker(client, "market.trades", worker_sql, batch_size=1)
    await worker.add(
        {
            "ts": dt.datetime.utcnow(),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "px": 100.0,
            "qty": 1.0,
            "side": "buy",
            "trade_id": "1",
        }
    )

    rows = await client.fetch("SELECT exchange, symbol, px, qty FROM market.trades")
    assert rows[0]["exchange"] == "binance"
    assert rows[0]["symbol"] == "BTCUSDT"
    assert float(rows[0]["px"]) == 100.0
    assert float(rows[0]["qty"]) == 1.0


@pytest.mark.asyncio
async def test_insert_and_query_bars(client):
    sql = """
        INSERT INTO market.bars (ts, timeframe, exchange, symbol, o, h, l, c, v)
        VALUES (:ts, :timeframe, :exchange, :symbol, :o, :h, :l, :c, :v)
    """
    from tradingbot.workers import BatchIngestionWorker

    worker = BatchIngestionWorker(client, "market.bars", sql, batch_size=1)
    await worker.add(
        {
            "ts": dt.datetime.utcnow(),
            "timeframe": "1m",
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "o": 100,
            "h": 110,
            "l": 90,
            "c": 105,
            "v": 10,
        }
    )

    rows = await client.fetch(
        "SELECT timeframe, o, c, v FROM market.bars WHERE symbol='BTCUSDT'"
    )
    assert rows[0]["timeframe"] == "1m"
    assert float(rows[0]["o"]) == 100
    assert float(rows[0]["c"]) == 105
    assert float(rows[0]["v"]) == 10


@pytest.mark.asyncio
async def test_insert_and_query_orderbook(client):
    sql = """
        INSERT INTO market.orderbook (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
        VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
    """
    from tradingbot.workers import BatchIngestionWorker

    worker = BatchIngestionWorker(client, "market.orderbook", sql, batch_size=1)
    await worker.add(
        {
            "ts": dt.datetime.utcnow(),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "bid_px": [100.0, 99.5],
            "bid_qty": [1.0, 0.5],
            "ask_px": [100.5, 101.0],
            "ask_qty": [0.8, 1.2],
        }
    )

    rows = await client.fetch(
        "SELECT bid_px, ask_px FROM market.orderbook WHERE symbol='BTCUSDT'"
    )
    assert rows[0]["bid_px"] == [100.0, 99.5]
    assert rows[0]["ask_px"] == [100.5, 101.0]


@pytest.mark.asyncio
async def test_insert_and_query_funding(client):
    sql = """
        INSERT INTO market.funding (ts, exchange, symbol, rate, interval_sec)
        VALUES (:ts, :exchange, :symbol, :rate, :interval_sec)
    """
    from tradingbot.workers import BatchIngestionWorker

    worker = BatchIngestionWorker(client, "market.funding", sql, batch_size=1)
    await worker.add(
        {
            "ts": dt.datetime.utcnow(),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "rate": 0.001,
            "interval_sec": 3600,
        }
    )

    rows = await client.fetch(
        "SELECT rate, interval_sec FROM market.funding WHERE symbol='BTCUSDT'"
    )
    assert float(rows[0]["rate"]) == 0.001
    assert rows[0]["interval_sec"] == 3600
