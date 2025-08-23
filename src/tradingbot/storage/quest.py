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


def insert_bba(
    engine,
    *,
    ts,
    exchange: str,
    symbol: str,
    bid_px: float | None,
    bid_qty: float | None,
    ask_px: float | None,
    ask_qty: float | None,
):
    """Persist a best bid/ask snapshot into QuestDB."""

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO bba (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
                VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
                """
            ),
            dict(
                ts=ts,
                exchange=exchange,
                symbol=symbol,
                bid_px=bid_px,
                bid_qty=bid_qty,
                ask_px=ask_px,
                ask_qty=ask_qty,
            ),
        )


def insert_book_delta(
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
    """Persist order book delta updates into QuestDB."""

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
                INSERT INTO book_delta (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
                VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
                """
            ),
            payload,
        )


def insert_bar_1m(engine, exchange: str, symbol: str, ts, o: float, h: float,
                  l: float, c: float, v: float):
    """Insert a 1-minute bar into QuestDB."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO bars (ts, timeframe, exchange, symbol, o, h, l, c, v)
                VALUES (:ts, '1m', :exchange, :symbol, :o, :h, :l, :c, :v)
                """
            ),
            dict(ts=ts, exchange=exchange, symbol=symbol, o=o, h=h, l=l, c=c, v=v),
        )


def insert_funding(engine, *, ts, exchange: str, symbol: str, rate: float, interval_sec: int):
    """Persist a funding rate record into QuestDB."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO funding (ts, exchange, symbol, rate, interval_sec)
                VALUES (:ts, :exchange, :symbol, :rate, :interval_sec)
                """
            ),
            dict(ts=ts, exchange=exchange, symbol=symbol, rate=rate, interval_sec=interval_sec),
        )


def insert_open_interest(engine, *, ts, exchange: str, symbol: str, oi: float):
    """Persist an open interest observation into QuestDB."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO open_interest (ts, exchange, symbol, oi)
                VALUES (:ts, :exchange, :symbol, :oi)
                """
            ),
            dict(ts=ts, exchange=exchange, symbol=symbol, oi=oi),
        )


def insert_basis(engine, *, ts, exchange: str, symbol: str, basis: float):
    """Persist a basis observation into QuestDB."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO basis (ts, exchange, symbol, basis)
                VALUES (:ts, :exchange, :symbol, :basis)
                """
            ),
            dict(ts=ts, exchange=exchange, symbol=symbol, basis=basis),
        )


def insert_order(
    engine,
    *,
    strategy: str,
    exchange: str,
    symbol: str,
    side: str,
    type_: str,
    qty: float,
    px: float | None,
    status: str,
    ext_order_id: str | None = None,
    notes: dict | None = None,
):
    """Persist an order record into QuestDB."""
    payload = dict(
        strategy=strategy,
        exchange=exchange,
        symbol=symbol,
        side=side,
        type=type_,
        qty=qty,
        px=px,
        status=status,
        ext_order_id=ext_order_id,
        notes=json.dumps(notes) if notes is not None else None,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO orders (strategy, exchange, symbol, side, type, qty, px, status, ext_order_id, notes)
                VALUES (:strategy, :exchange, :symbol, :side, :type, :qty, :px, :status, :ext_order_id, :notes)
                """
            ),
            payload,
        )


def insert_tri_signal(
    engine,
    *,
    exchange: str,
    base: str,
    mid: str,
    quote: str,
    direction: str,
    edge: float,
    notional_quote: float,
    taker_fee_bps: float,
    buffer_bps: float,
    bq: float,
    mq: float,
    mb: float,
):
    """Persist a triangular arbitrage signal into QuestDB."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO tri_signals (exchange, base, mid, quote, direction, edge, notional_quote, taker_fee_bps, buffer_bps, bq, mq, mb)
                VALUES (:exchange, :base, :mid, :quote, :direction, :edge, :notional_quote, :taker_fee_bps, :buffer_bps, :bq, :mq, :mb)
                """
            ),
            dict(
                exchange=exchange,
                base=base,
                mid=mid,
                quote=quote,
                direction=direction,
                edge=edge,
                notional_quote=notional_quote,
                taker_fee_bps=taker_fee_bps,
                buffer_bps=buffer_bps,
                bq=bq,
                mq=mq,
                mb=mb,
            ),
        )


def insert_cross_signal(
    engine,
    *,
    symbol: str,
    spot_exchange: str,
    perp_exchange: str,
    spot_px: float,
    perp_px: float,
    edge: float,
):
    """Persist a cross-exchange arbitrage signal into QuestDB."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO cross_signals (symbol, spot_exchange, perp_exchange, spot_px, perp_px, edge)
                VALUES (:symbol, :spot_exchange, :perp_exchange, :spot_px, :perp_px, :edge)
                """
            ),
            dict(
                symbol=symbol,
                spot_exchange=spot_exchange,
                perp_exchange=perp_exchange,
                spot_px=spot_px,
                perp_px=perp_px,
                edge=edge,
            ),
        )


__all__ = [
    "get_engine",
    "insert_trade",
    "insert_orderbook",
    "insert_bba",
    "insert_book_delta",
    "insert_bar_1m",
    "insert_funding",
    "insert_open_interest",
    "insert_basis",
    "insert_order",
    "insert_tri_signal",
    "insert_cross_signal",
]

