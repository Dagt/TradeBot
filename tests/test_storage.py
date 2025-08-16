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


def _setup_ts_engine():
    """SQLite engine with schemas for Timescale-style functions."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS market")
        conn.execute(
            text(
                """
                CREATE TABLE market.orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    side TEXT,
                    type TEXT,
                    qty REAL,
                    px REAL,
                    status TEXT,
                    ext_order_id TEXT,
                    notes TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE market.risk_events (
                    ts TEXT,
                    venue TEXT,
                    symbol TEXT,
                    kind TEXT,
                    message TEXT,
                    details TEXT
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


def test_insert_order_timescale():
    from tradingbot.storage import timescale

    engine = _setup_ts_engine()
    timescale.insert_order(
        engine,
        strategy="s",
        exchange="binance",
        symbol="BTCUSDT",
        side="buy",
        type_="limit",
        qty=1.0,
        px=100.0,
        status="filled",
    )

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT strategy, exchange, symbol, side, type, qty, px, status FROM market.orders"
            )
        ).fetchone()

    assert row == (
        "s",
        "binance",
        "BTCUSDT",
        "buy",
        "limit",
        1.0,
        100.0,
        "filled",
    )


def test_insert_risk_event_timescale():
    from tradingbot.storage import timescale

    engine = _setup_ts_engine()
    timescale.insert_risk_event(
        engine,
        venue="binance",
        symbol="",
        kind="daily_max_loss",
        message="",
        details="{}",
    )

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT venue, kind FROM market.risk_events")
        ).fetchone()

    assert row == ("binance", "daily_max_loss")

