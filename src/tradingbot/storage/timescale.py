import logging
from typing import Any, Iterable
from sqlalchemy import create_engine, text
from ..config import settings

log = logging.getLogger(__name__)

def pg_url() -> str:
    return f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"

def get_engine():
    return create_engine(pg_url(), pool_pre_ping=True)

def insert_trade(engine, t):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.trades (ts, exchange, symbol, px, qty, side, trade_id)
            VALUES (now(), :exchange, :symbol, :px, :qty, :side, :trade_id)
        '''), dict(exchange=t.exchange, symbol=t.symbol, px=t.price, qty=t.qty, side=t.side, trade_id=None))

def insert_bar_1m(engine, exchange: str, symbol: str, ts, o: float, h: float, l: float, c: float, v: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.bars (ts, timeframe, exchange, symbol, o, h, l, c, v)
            VALUES (:ts, '1m', :exchange, :symbol, :o, :h, :l, :c, :v)
        '''), dict(ts=ts, exchange=exchange, symbol=symbol, o=o, h=h, l=l, c=c, v=v))

def insert_order(engine, *, strategy: str, exchange: str, symbol: str, side: str, type_: str, qty: float, px: float | None, status: str, ext_order_id: str | None = None, notes: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.orders (strategy, exchange, symbol, side, type, qty, px, status, ext_order_id, notes)
            VALUES (:strategy, :exchange, :symbol, :side, :type, :qty, :px, :status, :ext_order_id, :notes)
        '''), dict(strategy=strategy, exchange=exchange, symbol=symbol, side=side, type=type_, qty=qty, px=px, status=status, ext_order_id=ext_order_id, notes=notes))

def insert_tri_signal(engine, *, exchange: str, base: str, mid: str, quote: str, direction: str, edge: float, notional_quote: float, taker_fee_bps: float, buffer_bps: float, bq: float, mq: float, mb: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.tri_signals (exchange, base, mid, quote, direction, edge, notional_quote, taker_fee_bps, buffer_bps, bq, mq, mb)
            VALUES (:exchange, :base, :mid, :quote, :direction, :edge, :notional_quote, :taker_fee_bps, :buffer_bps, :bq, :mq, :mb)
        '''), dict(exchange=exchange, base=base, mid=mid, quote=quote, direction=direction, edge=edge, notional_quote=notional_quote, taker_fee_bps=taker_fee_bps, buffer_bps=buffer_bps, bq=bq, mq=mq, mb=mb))

# --- Lecturas simples para API ---

def select_recent_tri_signals(engine, limit: int = 100) -> list[dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(text('''
            SELECT ts, exchange, base, mid, quote, direction, edge, notional_quote, taker_fee_bps, buffer_bps, bq, mq, mb
            FROM market.tri_signals
            ORDER BY ts DESC
            LIMIT :lim
        '''), dict(lim=limit)).mappings().all()
        return [dict(r) for r in rows]

def select_recent_orders(engine, limit: int = 100) -> list[dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(text('''
            SELECT ts, strategy, exchange, symbol, side, type, qty, px, status, ext_order_id, notes
            FROM market.orders
            ORDER BY ts DESC
            LIMIT :lim
        '''), dict(lim=limit)).mappings().all()
        return [dict(r) for r in rows]
