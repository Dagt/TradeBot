import logging
from typing import Any, Iterable
from sqlalchemy import create_engine, text
from ..config import settings
from . import timescale as _ts

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

from typing import Dict

def insert_portfolio_snapshot(engine, *, venue: str, symbol: str, position: float, price: float, notional_usd: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.portfolio_snapshots (venue, symbol, position, price, notional_usd)
            VALUES (:venue, :symbol, :position, :price, :notional_usd)
        '''), dict(venue=venue, symbol=symbol, position=position, price=price, notional_usd=notional_usd))

def select_latest_portfolio(engine, venue: str, limit_per_symbol: int = 1) -> list[dict]:
    # Trae la última fila por símbolo para un venue
    with engine.begin() as conn:
        rows = conn.execute(text('''
            SELECT DISTINCT ON (symbol) symbol, ts, position, price, notional_usd
            FROM market.portfolio_snapshots
            WHERE venue = :venue
            ORDER BY symbol, ts DESC
        '''), dict(venue=venue)).mappings().all()
        return [dict(r) for r in rows]

def insert_risk_event(engine, *, venue: str, symbol: str, kind: str, message: str, details: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.risk_events (venue, symbol, kind, message, details)
            VALUES (:venue, :symbol, :kind, :message, :details)
        '''), dict(venue=venue, symbol=symbol, kind=kind, message=message, details=details or {}))

def select_recent_risk_events(engine, venue: str, limit: int = 50) -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(text('''
            SELECT ts, venue, symbol, kind, message, details
            FROM market.risk_events
            WHERE venue = :venue
            ORDER BY ts DESC
            LIMIT :lim
        '''), dict(venue=venue, lim=limit)).mappings().all()
        return [dict(r) for r in rows]

def upsert_position(engine, *, venue: str, symbol: str, qty: float, avg_price: float, realized_pnl: float, fees_paid: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.positions (venue, symbol, qty, avg_price, realized_pnl, fees_paid)
            VALUES (:venue, :symbol, :qty, :avg_price, :realized_pnl, :fees_paid)
            ON CONFLICT (venue, symbol) DO UPDATE
            SET qty = EXCLUDED.qty,
                avg_price = EXCLUDED.avg_price,
                realized_pnl = EXCLUDED.realized_pnl,
                fees_paid = EXCLUDED.fees_paid
        '''), dict(venue=venue, symbol=symbol, qty=qty, avg_price=avg_price, realized_pnl=realized_pnl, fees_paid=fees_paid))

def insert_pnl_snapshot(engine, *, venue: str, symbol: str, qty: float, price: float, avg_price: float, upnl: float, rpnl: float, fees: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.pnl_snapshots (venue, symbol, qty, price, avg_price, upnl, rpnl, fees)
            VALUES (:venue, :symbol, :qty, :price, :avg_price, :upnl, :rpnl, :fees)
        '''), dict(venue=venue, symbol=symbol, qty=qty, price=price, avg_price=avg_price, upnl=upnl, rpnl=rpnl, fees=fees))

def select_pnl_summary(engine, venue: str) -> dict:
    with engine.begin() as conn:
        # última posición por símbolo
        rows = conn.execute(text('''
            SELECT p.symbol, p.qty, p.avg_price, p.realized_pnl, p.fees_paid
            FROM market.positions p
            WHERE p.venue = :venue
        '''), dict(venue=venue)).mappings().all()
        pos = [dict(r) for r in rows]

        # último mark por símbolo desde portfolio_snapshots (ya lo tomamos como ref de mark_px)
        marks = conn.execute(text('''
            SELECT DISTINCT ON (symbol) symbol, price
            FROM market.portfolio_snapshots
            WHERE venue = :venue
            ORDER BY symbol, ts DESC
        '''), dict(venue=venue)).mappings().all()
        mark_map = {r["symbol"]: float(r["price"]) for r in marks}

        for r in pos:
            mp = mark_map.get(r["symbol"])
            if mp is None:
                r["upnl"] = 0.0
            else:
                # (mark - avg)*qty
                r["upnl"] = (mp - float(r["avg_price"])) * float(r["qty"])
            r["total_pnl"] = r["upnl"] + float(r["realized_pnl"]) - float(r["fees_paid"])
        totals = {
            "upnl": sum(float(x["upnl"]) for x in pos),
            "rpnl": sum(float(x["realized_pnl"]) for x in pos),
            "fees": sum(float(x["fees_paid"]) for x in pos),
        }
        totals["net_pnl"] = totals["upnl"] + totals["rpnl"] - totals["fees"]
        return {"items": pos, "totals": totals}
