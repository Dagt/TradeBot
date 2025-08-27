import logging
from typing import Any, Iterable
from types import SimpleNamespace
from datetime import datetime

from sqlalchemy import create_engine, text

from ..config import settings
from ..core.symbols import normalize

log = logging.getLogger(__name__)

def pg_url() -> str:
    return f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"

def get_engine():
    """Return a SQLAlchemy engine if TimescaleDB is reachable.

    By default ``create_engine`` does not actually establish a connection.
    This caused the API to believe TimescaleDB was available even when the
    database server was down, leading to runtime ``OperationalError``
    exceptions for each request.  Here we try a simple ``SELECT 1`` query
    during initialisation; if it fails we propagate the exception so the API
    can disable database-backed endpoints gracefully.
    """

    engine = create_engine(pg_url(), pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        # Bubble up the exception so callers can handle the absence of the DB
        # (e.g. by disabling Timescale-backed features).
        raise
    return engine

def insert_trades(engine, trades: Iterable[Any]):
    """Insert multiple trade ticks using a single batch statement."""

    payload = [
        dict(
            ts=t.ts,
            exchange=t.exchange,
            symbol=normalize(getattr(t, "symbol", "")),
            px=t.price,
            qty=t.qty,
            side=t.side,
            trade_id=getattr(t, "trade_id", None),
        )
        for t in trades
    ]
    if not payload:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO market.trades (ts, exchange, symbol, px, qty, side, trade_id)
            VALUES (:ts, :exchange, :symbol, :px, :qty, :side, :trade_id)
            """
            ),
            payload,
        )


def insert_trade(engine, t):
    insert_trades(engine, [t])

def insert_bars_1m(engine, bars: Iterable[Any]):
    """Insert multiple 1-minute OHLCV bars."""

    payload = [
        dict(
            ts=b.ts,
            exchange=b.exchange,
            symbol=normalize(getattr(b, "symbol", "")),
            o=b.o,
            h=b.h,
            l=b.l,
            c=b.c,
            v=b.v,
        )
        for b in bars
    ]
    if not payload:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO market.bars (ts, timeframe, exchange, symbol, o, h, l, c, v)
            VALUES (:ts, '1m', :exchange, :symbol, :o, :h, :l, :c, :v)
            """
            ),
            payload,
        )


def insert_bar_1m(engine, exchange: str, symbol: str, ts, o: float, h: float,
                  low: float, c: float, v: float):
    """Insert a single 1-minute OHLCV bar into TimescaleDB."""

    bar = SimpleNamespace(
        ts=ts,
        exchange=exchange,
        symbol=normalize(symbol),
        o=o,
        h=h,
        l=low,
        c=c,
        v=v,
    )
    insert_bars_1m(engine, [bar])

def insert_funding(
    engine,
    *,
    ts,
    exchange: str,
    symbol: str,
    rate: float,
    interval_sec: int,
):
    """Persist a funding rate observation."""

    with engine.begin() as conn:
        conn.execute(
            text(
                '''
            INSERT INTO market.funding (ts, exchange, symbol, rate, interval_sec)
            VALUES (:ts, :exchange, :symbol, :rate, :interval_sec)
        '''
            ),
            dict(
                ts=ts,
                exchange=exchange,
                symbol=normalize(symbol),
                rate=rate,
                interval_sec=interval_sec,
            ),
        )


def insert_open_interest(engine, *, ts, exchange: str, symbol: str, oi: float):
    """Persist open interest readings."""
    with engine.begin() as conn:
        conn.execute(
            text(
                '''
            INSERT INTO market.open_interest (ts, exchange, symbol, oi)
            VALUES (:ts, :exchange, :symbol, :oi)
        '''
            ),
            dict(ts=ts, exchange=exchange, symbol=normalize(symbol), oi=oi),
        )


def insert_basis(engine, *, ts, exchange: str, symbol: str, basis: float):
    """Persist basis readings."""
    with engine.begin() as conn:
        conn.execute(
            text(
                '''
            INSERT INTO market.basis (ts, exchange, symbol, basis)
            VALUES (:ts, :exchange, :symbol, :basis)
        '''
            ),
            dict(ts=ts, exchange=exchange, symbol=normalize(symbol), basis=basis),
        )

def insert_orderbook(engine, *, ts, exchange: str, symbol: str,
                     bid_px: list[float], bid_qty: list[float],
                     ask_px: list[float], ask_qty: list[float]):
    with engine.begin() as conn:
        conn.execute(
            text('''
            INSERT INTO market.orderbook (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
            VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
        '''),
            dict(
                ts=ts,
                exchange=exchange,
                symbol=normalize(symbol),
                bid_px=bid_px,
                bid_qty=bid_qty,
                ask_px=ask_px,
                ask_qty=ask_qty,
            ),
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
    """Persist best bid/ask snapshots."""

    with engine.begin() as conn:
        conn.execute(
            text(
                '''
            INSERT INTO market.bba (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
            VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
        '''
            ),
            dict(
                ts=ts,
                exchange=exchange,
                symbol=normalize(symbol),
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
    """Persist order book delta updates."""

    with engine.begin() as conn:
        conn.execute(
            text(
                '''
            INSERT INTO market.book_delta (ts, exchange, symbol, bid_px, bid_qty, ask_px, ask_qty)
            VALUES (:ts, :exchange, :symbol, :bid_px, :bid_qty, :ask_px, :ask_qty)
        '''
            ),
            dict(
                ts=ts,
                exchange=exchange,
                symbol=normalize(symbol),
                bid_px=bid_px,
                bid_qty=bid_qty,
                ask_px=ask_px,
                ask_qty=ask_qty,
            ),
        )

def insert_order(engine, *, strategy: str, exchange: str, symbol: str, side: str, type_: str, qty: float, px: float | None, status: str, ext_order_id: str | None = None, notes: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.orders (strategy, exchange, symbol, side, type, qty, px, status, ext_order_id, notes)
            VALUES (:strategy, :exchange, :symbol, :side, :type, :qty, :px, :status, :ext_order_id, :notes)
        '''), dict(strategy=strategy, exchange=exchange, symbol=normalize(symbol), side=side, type=type_, qty=qty, px=px, status=status, ext_order_id=ext_order_id, notes=notes))

def insert_tri_signal(engine, *, exchange: str, base: str, mid: str, quote: str, direction: str, edge: float, notional: float, taker_fee_bps: float, buffer_bps: float, bq: float, mq: float, mb: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.tri_signals (exchange, base, mid, quote, direction, edge, notional_quote, taker_fee_bps, buffer_bps, bq, mq, mb)
            VALUES (:exchange, :base, :mid, :quote, :direction, :edge, :notional, :taker_fee_bps, :buffer_bps, :bq, :mq, :mb)
        '''), dict(exchange=exchange, base=base, mid=mid, quote=quote, direction=direction, edge=edge, notional=notional, taker_fee_bps=taker_fee_bps, buffer_bps=buffer_bps, bq=bq, mq=mq, mb=mb))


def insert_cross_signal(engine, *, symbol: str, spot_exchange: str, perp_exchange: str,
                        spot_px: float, perp_px: float, edge: float):
    """Persist cross-exchange arbitrage opportunities."""
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.cross_signals (symbol, spot_exchange, perp_exchange, spot_px, perp_px, edge)
            VALUES (:symbol, :spot_exchange, :perp_exchange, :spot_px, :perp_px, :edge)
        '''), dict(symbol=normalize(symbol), spot_exchange=spot_exchange, perp_exchange=perp_exchange,
                   spot_px=spot_px, perp_px=perp_px, edge=edge))

def select_bars(
    engine,
    *,
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1m",
) -> list[dict[str, Any]]:
    """Load OHLCV bars from TimescaleDB."""
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT ts, o, h, l, c, v
            FROM market.bars
            WHERE exchange = :exchange AND symbol = :symbol
              AND timeframe = :tf AND ts BETWEEN :start AND :end
            ORDER BY ts
            """
                ),
                dict(exchange=exchange, symbol=symbol, tf=timeframe, start=start, end=end),
            )
            .mappings()
            .all()
        )
        return [dict(r) for r in rows]

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


def select_order_history(
    engine,
    *,
    limit: int = 100,
    search: str | None = None,
    symbol: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    query = [
        "SELECT ts, strategy, exchange, symbol, side, type, qty, px, status, ext_order_id, notes",
        "FROM market.orders",
    ]
    params: dict[str, Any] = {"lim": limit}
    conditions: list[str] = []
    if symbol:
        conditions.append("symbol ILIKE :sym")
        params["sym"] = f"%{symbol}%"
    if status:
        conditions.append("status ILIKE :stat")
        params["stat"] = f"%{status}%"
    if search:
        conditions.append("(symbol ILIKE :search OR strategy ILIKE :search OR status ILIKE :search)")
        params["search"] = f"%{search}%"
    if conditions:
        query.append("WHERE " + " AND ".join(conditions))
    query.append("ORDER BY ts DESC")
    query.append("LIMIT :lim")
    sql = "\n".join(query)
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]

def insert_portfolio_snapshot(engine, *, venue: str, symbol: str, position: float, price: float, notional_usd: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.portfolio_snapshots (venue, symbol, position, price, notional_usd)
            VALUES (:venue, :symbol, :position, :price, :notional_usd)
        '''), dict(venue=venue, symbol=normalize(symbol), position=position, price=price, notional_usd=notional_usd))

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
        '''), dict(venue=venue, symbol=normalize(symbol), kind=kind, message=message, details=details or {}))

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
        '''), dict(venue=venue, symbol=normalize(symbol), qty=qty, avg_price=avg_price, realized_pnl=realized_pnl, fees_paid=fees_paid))

def insert_pnl_snapshot(engine, *, venue: str, symbol: str, qty: float, price: float, avg_price: float, upnl: float, rpnl: float, fees: float):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.pnl_snapshots (venue, symbol, qty, price, avg_price, upnl, rpnl, fees)
            VALUES (:venue, :symbol, :qty, :price, :avg_price, :upnl, :rpnl, :fees)
        '''), dict(venue=venue, symbol=normalize(symbol), qty=qty, price=price, avg_price=avg_price, upnl=upnl, rpnl=rpnl, fees=fees))

def select_pnl_summary(engine, venue: str, symbol: str | None = None) -> dict:
    with engine.begin() as conn:
        where = "p.venue = :venue"
        params = {"venue": venue}
        if symbol:
            where += " AND p.symbol = :symbol"
            params["symbol"] = symbol
        rows = conn.execute(
            text(f'''
            SELECT p.symbol, p.qty, p.avg_price, p.realized_pnl, p.fees_paid
            FROM market.positions p
            WHERE {where}
        '''),
            params,
        ).mappings().all()
        pos = [dict(r) for r in rows]

        # último mark por símbolo desde portfolio_snapshots (ya lo tomamos como ref de mark_px)
        mark_rows = conn.execute(
            text(
                f'''
            SELECT DISTINCT ON (symbol) symbol, price
            FROM market.portfolio_snapshots
            WHERE venue = :venue {"AND symbol = :symbol" if symbol else ""}
            ORDER BY symbol, ts DESC
        '''
            ),
            params,
        ).mappings().all()
        mark_map = {r["symbol"]: float(r["price"]) for r in mark_rows}

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

# --- Rehidratación de posiciones y marks ---

def load_positions(engine, venue: str) -> dict:
    """Devuelve {symbol: {'qty': float, 'avg_price': float, 'realized_pnl': float, 'fees_paid': float}}"""
    with engine.begin() as conn:
        rows = conn.execute(text('''
            SELECT venue, symbol, qty, avg_price, realized_pnl, fees_paid
            FROM market.positions
            WHERE venue = :venue
        '''), dict(venue=venue)).mappings().all()
        out = {}
        for r in rows:
            out[r["symbol"]] = {
                "qty": float(r["qty"]),
                "avg_price": float(r["avg_price"]),
                "realized_pnl": float(r["realized_pnl"]),
                "fees_paid": float(r["fees_paid"]),
            }
        return out

def load_latest_marks(engine, venue: str) -> dict:
    """Devuelve {symbol: price} a partir de portfolio_snapshots (último precio conocido por símbolo)."""
    with engine.begin() as conn:
        rows = conn.execute(text('''
            SELECT DISTINCT ON (symbol) symbol, price
            FROM market.portfolio_snapshots
            WHERE venue = :venue
            ORDER BY symbol, ts DESC
        '''), dict(venue=venue)).mappings().all()
        return {r["symbol"]: float(r["price"]) for r in rows}

# --- PnL timeseries (agrupado por time_bucket) ---
def select_pnl_timeseries(engine, *, venue: str, symbol: str | None = None, bucket: str = "1 minute", hours: int = 6) -> list[dict]:
    """
    Devuelve puntos agregados por time_bucket en las últimas `hours` horas.
    Si symbol es None => agrega todos los símbolos del venue.
    net = sum(upnl + rpnl - fees) por bucket.
    """
    where = "venue = :venue AND ts >= now() - (:hours::text || ' hours')::interval"
    params = {"venue": venue, "hours": hours, "bucket": bucket}
    if symbol:
        where += " AND symbol = :symbol"
        params["symbol"] = symbol

    sql = f"""
        SELECT
          time_bucket(:bucket, ts) AS bucket_ts,
          SUM(upnl)  AS upnl_sum,
          SUM(rpnl)  AS rpnl_sum,
          SUM(fees)  AS fees_sum,
          SUM(upnl + rpnl - fees) AS net_sum
        FROM market.pnl_snapshots
        WHERE {where}
        GROUP BY bucket_ts
        ORDER BY bucket_ts ASC
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        out = []
        for r in rows:
            out.append({
                "ts": r["bucket_ts"].isoformat(),
                "upnl": float(r["upnl_sum"] or 0.0),
                "rpnl": float(r["rpnl_sum"] or 0.0),
                "fees": float(r["fees_sum"] or 0.0),
                "net": float(r["net_sum"] or 0.0),
            })
        return out

# --- Fills: inserción y consulta ---
def insert_fill(engine, *, ts, venue: str, strategy: str, symbol: str, side: str, type_: str,
                qty: float, price: float | None, fee_usdt: float | None,
                reduce_only: bool = False, ext_order_id: str | None = None, raw: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.fills (ts, venue, strategy, symbol, side, type, qty, price, notional, fee_usdt, reduce_only, ext_order_id, raw)
            VALUES (:ts, :venue, :strategy, :symbol, :side, :type, :qty, :price,
                    CASE WHEN :price IS NULL THEN NULL ELSE :qty * :price END,
                    :fee_usdt, :reduce_only, :ext_order_id, :raw)
        '''), dict(
            ts=ts,
            venue=venue,
            strategy=strategy,
            symbol=normalize(symbol),
            side=side,
            type=type_,
            qty=float(qty),
            price=(float(price) if price is not None else None),
            fee_usdt=(float(fee_usdt) if fee_usdt is not None else None),
            reduce_only=bool(reduce_only),
            ext_order_id=ext_order_id,
            raw=raw or {},
        ))

def select_recent_fills(engine, venue: str, symbol: str | None = None, limit: int = 100) -> list[dict]:
    where = "venue = :venue"
    params = {"venue": venue, "lim": limit}
    if symbol:
        where += " AND symbol = :symbol"
        params["symbol"] = symbol
    sql = f'''
      SELECT ts, strategy, symbol, side, type, qty, price, notional, fee_usdt, reduce_only, ext_order_id
      FROM market.fills
      WHERE {where}
      ORDER BY ts DESC
      LIMIT :lim
    '''
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]

def select_fills_summary(engine, venue: str, symbol: str | None = None, strategy: str | None = None) -> list[dict]:
    where = "venue = :venue"
    params = {"venue": venue}
    if symbol:
        where += " AND symbol = :symbol"
        params["symbol"] = symbol
    if strategy:
        where += " AND strategy = :strategy"
        params["strategy"] = strategy
    sql = f'''
      SELECT strategy, symbol, COUNT(*) AS fills,
             SUM(COALESCE(notional,0)) AS notional,
             SUM(COALESCE(fee_usdt,0)) AS fees
      FROM market.fills
      WHERE {where}
      GROUP BY strategy, symbol
      ORDER BY strategy, symbol
    '''
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]

# --- Conciliación: reconstruir posiciones desde fills (opcional) ---
def rebuild_positions_from_fills(engine, venue: str) -> dict:
    """
    Recorre fills históricos y reconstruye qty/avg_price/rpnl/fees por símbolo.
    Devuelve {symbol: {...}}. No escribe; solo calcula.
    """
    # Traemos todos los fills del venue ordenados cronológicamente
    sql = '''
      SELECT ts, symbol, side, type, qty, price, fee_usdt
      FROM market.fills
      WHERE venue = :venue AND price IS NOT NULL
      ORDER BY ts ASC
    '''
    with engine.begin() as conn:
        rows = conn.execute(text(sql), dict(venue=venue)).mappings().all()

    from ..risk.pnl import Position, apply_fill
    out: dict[str, Position] = {}
    # Asumimos fee bps aproximadas; si tus fills traen fee_usdt real, lo sumaremos aparte
    fee_bps_guess = 2.0 if "futures" in venue else 7.5
    for r in rows:
        sym = r["symbol"]
        if sym not in out:
            out[sym] = Position()
        p = out[sym]
        px = float(r["price"])
        q = float(r["qty"])
        side = (r["side"] or "buy").lower()
        p = apply_fill(p, side, q, px, fee_bps_guess, venue_type=("futures" if "futures" in venue else "spot"))
        # ajusta fees si viene real
        fee = r["fee_usdt"]
        if fee is not None:
            p.fees_paid += float(fee)
        out[sym] = p
    # normalizamos a dict serializable
    return {k: dict(qty=v.qty, avg_price=v.avg_price, realized_pnl=v.realized_pnl, fees_paid=v.fees_paid) for k, v in out.items()}

# --- Slippage agregado desde fills ---
def select_slippage(engine, *, venue: str, symbol: str | None = None, hours: int = 6):
    """
    Slippage = (fill_price - ref_price) / ref_price.
    Usamos ref_price = último price en portfolio_snapshots anterior o cercano al ts del fill.
    Si no hay snapshot cercano, omitimos el fill para slippage.
    Devuelve agregados: count, mean, p50, p90, p99 por lado y global.
    """
    # 1) Traer fills recientes con price
    sql_fills = '''
      SELECT ts, symbol, side, price
      FROM market.fills
      WHERE venue = :venue
        AND ts >= now() - (:hours::text || ' hours')::interval
        AND price IS NOT NULL
      ORDER BY ts ASC
    '''
    # 2) Para cada fill, buscamos el snapshot inmediatamente anterior (misma symbol)
    sql_ref = '''
      SELECT price FROM market.portfolio_snapshots
      WHERE venue = :venue AND symbol = :symbol AND ts <= :ts
      ORDER BY ts DESC
      LIMIT 1
    '''
    import numpy as np
    out = {"global": [], "buy": [], "sell": []}
    with engine.begin() as conn:
        fills = conn.execute(text(sql_fills), {"venue": venue, "hours": hours}).mappings().all()
        for f in fills:
            sym = f["symbol"]
            ts = f["ts"]
            px_fill = float(f["price"])
            ref_row = conn.execute(text(sql_ref), {"venue": venue, "symbol": sym, "ts": ts}).mappings().first()
            if not ref_row:
                continue
            px_ref = float(ref_row["price"])
            if px_ref <= 0:
                continue
            slip = (px_fill - px_ref) / px_ref
            out["global"].append(slip)
            out[f["side"].lower()].append(slip)

    def stats(arr):
        if not arr:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        a = np.array(arr, dtype=float)
        return {
            "count": int(a.size),
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
        }
    return {
        "venue": venue,
        "hours": hours,
        "global": stats(out["global"]),
        "buy": stats(out["buy"]),
        "sell": stats(out["sell"])
    }

# --- OCO persistence ---------------------------------------------------------
def upsert_oco(engine, *, venue: str, symbol: str, side: str,
               qty: float, entry_price: float, sl_price: float, tp_price: float, status: str = "active"):
    sql = text("""
    INSERT INTO market.oco_orders (venue, symbol, side, qty, entry_price, sl_price, tp_price, status)
    VALUES (:venue, :symbol, :side, :qty, :entry_price, :sl_price, :tp_price, :status)
    ON CONFLICT (id) DO NOTHING; -- no natural key, insert simple (dejamos 1 activo por símbolo)
    """)
    # Antes, cancelamos cualquiera activo para ese venue/símbolo (1 OCO activo por símbolo)
    sql_cancel_old = text("""
      UPDATE market.oco_orders
      SET status='cancelled', updated_at=now()
      WHERE venue=:venue AND symbol=:symbol AND status='active'
    """)
    norm_symbol = normalize(symbol)
    with engine.begin() as conn:
        conn.execute(sql_cancel_old, {"venue": venue, "symbol": norm_symbol})
        conn.execute(sql, {
            "venue": venue,
            "symbol": norm_symbol,
            "side": side,
            "qty": float(qty),
            "entry_price": float(entry_price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "status": status,
        })

def update_oco_active(engine, *, venue: str, symbol: str,
                      qty: float, entry_price: float, sl_price: float, tp_price: float):
    sql = text("""
      UPDATE market.oco_orders
      SET qty=:qty, entry_price=:entry_price, sl_price=:sl_price, tp_price=:tp_price, updated_at=now()
      WHERE venue=:venue AND symbol=:symbol AND status='active'
    """)
    norm_symbol = normalize(symbol)
    with engine.begin() as conn:
        conn.execute(sql, {
            "venue": venue,
            "symbol": norm_symbol,
            "qty": float(qty),
            "entry_price": float(entry_price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
        })

def mark_oco_triggered(engine, *, venue: str, symbol: str, triggered: str):
    sql = text("""
      UPDATE market.oco_orders
      SET status='triggered', triggered=:triggered, updated_at=now()
      WHERE venue=:venue AND symbol=:symbol AND status='active'
    """)
    norm_symbol = normalize(symbol)
    with engine.begin() as conn:
        conn.execute(sql, {"venue": venue, "symbol": norm_symbol, "triggered": triggered})

def cancel_oco(engine, *, venue: str, symbol: str):
    sql = text("""
      UPDATE market.oco_orders
      SET status='cancelled', updated_at=now()
      WHERE venue=:venue AND symbol=:symbol AND status='active'
    """)
    norm_symbol = normalize(symbol)
    with engine.begin() as conn:
        conn.execute(sql, {"venue": venue, "symbol": norm_symbol})

def load_active_oco_by_symbols(engine, *, venue: str, symbols: list[str]):
    if not symbols:
        return {}
    # simple IN; si prefieres, usa un temp table
    placeholders = ",".join([f":s{i}" for i in range(len(symbols))])
    params = {"venue": venue}
    for i, s in enumerate(symbols):
        params[f"s{i}"] = s
    sql = text(f"""
      SELECT venue, symbol, side, qty, entry_price, sl_price, tp_price
      FROM market.oco_orders
      WHERE venue=:venue AND status='active' AND symbol IN ({placeholders})
    """)
    out = {}
    with engine.begin() as conn:
        for row in conn.execute(sql, params).mappings():
            out[row["symbol"]] = {
                "side": row["side"],
                "qty": float(row["qty"]),
                "entry_price": float(row["entry_price"]),
                "sl_price": float(row["sl_price"]),
                "tp_price": float(row["tp_price"]),
            }
    return out
