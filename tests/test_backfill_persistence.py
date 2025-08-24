from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


SCHEMA_SQL = Path("db/schema.sql").read_text()

pytestmark = pytest.mark.integration

if not os.environ.get("PG_TEST", ""):
    pytest.skip("PostgreSQL not available", allow_module_level=True)


class DummyExchange:
    rateLimit = 0
    id = "dummy"

    def __init__(self) -> None:
        self._ohlcv_used = False
        self._trades_used = False

    async def fetch_ohlcv(self, symbol, timeframe, since, limit):  # noqa: ANN001
        if self._ohlcv_used:
            return []
        self._ohlcv_used = True
        # Two bars spaced one minute apart
        return [
            [since, 1.0, 2.0, 0.5, 1.5, 10.0],
            [since + 60_000, 1.1, 2.1, 0.6, 1.6, 11.0],
        ]

    async def fetch_trades(self, symbol, since, limit):  # noqa: ANN001
        if self._trades_used:
            return []
        self._trades_used = True
        return [
            {
                "timestamp": since,
                "price": 1.0,
                "amount": 2.0,
                "side": "buy",
                "id": str(since),
            },
            {
                "timestamp": since + 1,
                "price": 1.1,
                "amount": 2.1,
                "side": "sell",
                "id": str(since + 1),
            },
        ]

    async def close(self) -> None:  # pragma: no cover - nothing to do
        pass


@pytest_asyncio.fixture(scope="module")
async def setup_db():
    dsn = "postgresql+asyncpg://postgres:postgres@localhost/tradebot_test"
    eng = create_async_engine(dsn, echo=False)
    async with eng.begin() as conn:
        for stmt in SCHEMA_SQL.split(";"):
            s = stmt.strip()
            if s:
                await conn.execute(text(s))
    yield
    await eng.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("input_symbol", ["BTC/USDT", "BTC-USDT"])
async def test_backfill_persists_data(monkeypatch, setup_db, input_symbol):
    monkeypatch.setenv("PG_HOST", "localhost")
    monkeypatch.setenv("PG_USER", "postgres")
    monkeypatch.setenv("PG_PASSWORD", "postgres")
    monkeypatch.setenv("PG_DB", "tradebot_test")

    from tradingbot.jobs import backfill as job_backfill

    monkeypatch.setattr(job_backfill.ccxt, "binance", lambda *_, **__: DummyExchange())

    await job_backfill.backfill(days=1, symbols=[input_symbol])

    from tradingbot.storage.async_timescale import AsyncTimescaleClient

    client = AsyncTimescaleClient(
        dsn="postgresql+asyncpg://postgres:postgres@localhost/tradebot_test"
    )
    bars = await client.fetch("SELECT symbol, o, v FROM market.bars")
    trades = await client.fetch("SELECT symbol, px, qty FROM market.trades")
    await client.close()

    assert bars and bars[0]["symbol"] == "BTCUSDT"
    assert trades and trades[0]["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
@pytest.mark.parametrize("input_symbol", ["BTC/USDT", "BTC-USDT"])
async def test_backfill_overlapping_range(monkeypatch, setup_db, input_symbol):
    # Ensure database is empty
    dsn = "postgresql+asyncpg://postgres:postgres@localhost/tradebot_test"
    eng = create_async_engine(dsn, echo=False)
    async with eng.begin() as conn:
        await conn.execute(text("TRUNCATE market.bars"))
        await conn.execute(text("TRUNCATE market.trades"))
    await eng.dispose()

    monkeypatch.setenv("PG_HOST", "localhost")
    monkeypatch.setenv("PG_USER", "postgres")
    monkeypatch.setenv("PG_PASSWORD", "postgres")
    monkeypatch.setenv("PG_DB", "tradebot_test")

    from tradingbot.jobs import backfill as job_backfill

    monkeypatch.setattr(job_backfill.ccxt, "binance", lambda *_, **__: DummyExchange())

    start1 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end1 = start1 + timedelta(minutes=2)
    await job_backfill.backfill(days=1, symbols=[input_symbol], start=start1, end=end1)

    start2 = start1 + timedelta(minutes=1)
    end2 = start2 + timedelta(minutes=2)
    await job_backfill.backfill(days=1, symbols=[input_symbol], start=start2, end=end2)

    from tradingbot.storage.async_timescale import AsyncTimescaleClient

    client = AsyncTimescaleClient(dsn=dsn)
    bars = await client.fetch("SELECT count(*) AS cnt FROM market.bars")
    trades = await client.fetch("SELECT count(*) AS cnt FROM market.trades")
    await client.close()

    assert bars[0]["cnt"] == 3
    assert trades[0]["cnt"] == 3

