from __future__ import annotations

import json
import importlib
from types import SimpleNamespace

from sqlalchemy import create_engine, text

from tradingbot.config import settings


def _setup_engine():
    """Create an in-memory SQLite engine with minimal tables."""

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE trades (
                    ts TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    px REAL,
                    qty REAL,
                    side TEXT,
                    trade_id TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE orderbook (
                    ts TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    bid_px TEXT,
                    bid_qty TEXT,
                    ask_px TEXT,
                    ask_qty TEXT
                )
                """
            )
        )
    return engine


def _reload_storage_backend(backend: str):
    settings.db_backend = backend
    import tradingbot.storage as storage

    return importlib.reload(storage)


def test_insert_and_read_trade():
    storage = _reload_storage_backend("questdb")
    engine = _setup_engine()

    t = SimpleNamespace(exchange="binance", symbol="BTCUSDT", price=100.0, qty=1.0, side="buy")
    storage.insert_trade(engine, t)

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT exchange, symbol, px, qty, side FROM trades")
        ).fetchone()

    assert row == ("binance", "BTCUSDT", 100.0, 1.0, "buy")


def test_insert_and_read_orderbook():
    storage = _reload_storage_backend("questdb")
    engine = _setup_engine()

    storage.insert_orderbook(
        engine,
        ts="2024-01-01T00:00:00",
        exchange="binance",
        symbol="BTCUSDT",
        bid_px=[100.0, 99.5],
        bid_qty=[1.0, 0.5],
        ask_px=[100.5, 101.0],
        ask_qty=[0.8, 1.2],
    )

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT bid_px, bid_qty, ask_px, ask_qty FROM orderbook")
        ).fetchone()

    assert json.loads(row[0]) == [100.0, 99.5]
    assert json.loads(row[1]) == [1.0, 0.5]
    assert json.loads(row[2]) == [100.5, 101.0]
    assert json.loads(row[3]) == [0.8, 1.2]

