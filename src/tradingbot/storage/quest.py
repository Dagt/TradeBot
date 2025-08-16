"""Storage helpers for QuestDB.

This module mirrors a small subset of the TimescaleDB storage helpers so that
the rest of the application can switch between backends with minimal changes.
QuestDB speaks the PostgreSQL wire protocol, therefore we reuse SQLAlchemy with
the ``psycopg2`` driver to perform inserts.

The implementation here intentionally keeps the SQL very small and avoids
QuestDB specific extensions so that the functions are also easy to unit test
using an in-memory SQLite database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import create_engine, text

from ..config import settings

log = logging.getLogger(__name__)


def qs_url() -> str:
    """Build the SQLAlchemy URL for QuestDB.

    QuestDB exposes the PostgreSQL wire protocol on port 8812.  Credentials are
    optional in a default installation but are included here for completeness
    and to mirror the structure used for TimescaleDB.
    """

    return (
        f"postgresql+psycopg2://{settings.questdb_user}:{settings.questdb_password}"
        f"@{settings.questdb_host}:{settings.questdb_port}/{settings.questdb_db}"
    )


def get_engine():
    """Return a SQLAlchemy engine connected to QuestDB."""

    return create_engine(qs_url(), pool_pre_ping=True)


def insert_trade(engine, t):
    """Insert a trade tick into QuestDB.

    Parameters
    ----------
    engine: SQLAlchemy engine obtained via :func:`get_engine` or any compatible
        engine (e.g. an in-memory SQLite engine for tests).
    t: object
        Trade-like object with ``exchange``, ``symbol``, ``price`` (``px``),
        ``qty`` and ``side`` attributes.  A timestamp field is optional; if not
        provided the current UTC time is used.
    """

    ts = getattr(t, "ts", None) or datetime.utcnow()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trades (ts, exchange, symbol, px, qty, side, trade_id)
                VALUES (:ts, :exchange, :symbol, :px, :qty, :side, :trade_id)
                """
            ),
            dict(
                ts=ts,
                exchange=t.exchange,
                symbol=t.symbol,
                px=t.price,
                qty=t.qty,
                side=t.side,
                trade_id=getattr(t, "trade_id", None),
            ),
        )


def insert_orderbook(
    engine,
    *,
    ts,
    exchange: str,
    symbol: str,
    bid_px: list[float],
    bid_qty: list[float],
    ask_px: list[float],
    ask_qty: list[float],
):
    """Persist an orderbook snapshot into QuestDB.

    Lists are serialized to JSON before insertion so that the function works
    across different database engines (QuestDB stores them as ``STRING`` when
    inserted via the PostgreSQL interface).
    """

    payload = dict(
        ts=ts,
        exchange=exchange,
        symbol=symbol,
        bid_px=json.dumps(bid_px),
        bid_qty=json.dumps(bid_qty),
        ask_px=json.dumps(ask_px),
        ask_qty=json.dumps(ask_qty),
    )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO orderbook (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
                VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
                """
            ),
            payload,
        )


__all__ = ["get_engine", "insert_trade", "insert_orderbook"]

