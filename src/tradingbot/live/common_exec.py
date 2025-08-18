# src/tradingbot/live/common_exec.py
from __future__ import annotations
from datetime import datetime, timezone

from ..utils.logging import get_logger

log = get_logger(__name__)

try:
    from ..storage.timescale import (
        insert_bar_1m, insert_portfolio_snapshot, insert_order, insert_fill,
        upsert_position, insert_pnl_snapshot, insert_risk_event
    )
    _CAN_PG = True
except Exception:
    _CAN_PG = False


def persist_bar_and_snapshot(
    engine,
    venue: str,
    symbol: str,
    bar,  # objeto con ts_open, o, h, l, c, v
    cur_pos: float | None = None
):
    """
    Inserta barra 1m y snapshot de portafolio (si hay engine).
    """
    if engine is None or not _CAN_PG:
        return
    try:
        insert_bar_1m(
            engine, exchange=venue, symbol=symbol,
            ts=bar.ts_open, o=bar.o, h=bar.h, l=bar.l, c=bar.c, v=bar.v
        )
    except Exception:
        log.exception("[%s] persist failure: insert_bar_1m", symbol)

    if cur_pos is not None:
        try:
            insert_portfolio_snapshot(
                engine, venue=venue, symbol=symbol,
                position=cur_pos, price=bar.c, notional_usd=abs(cur_pos) * bar.c
            )
        except Exception:
            log.exception("[%s] persist failure: insert_portfolio_snapshot", symbol)


def persist_after_order(
    engine,
    *,
    venue: str,
    strategy: str,
    symbol: str,
    side: str,
    type_: str,
    qty: float,
    mark_price: float,
    resp: dict | None,
    reduce_only: bool,
    # Datos de posición/PnL después de aplicar el fill en memoria:
    pos_obj,      # objeto con qty, avg_price, realized_pnl, fees_paid
    pnl_mark_px: float | None = None,
    risk_event_kind: str | None = None,
    risk_event_message: str | None = None,
    risk_event_details: dict | None = None,
):
    """
    Inserta order + fill + upsert_position + pnl_snapshot y opcionalmente risk_event.
    Usa campos estándar de tu adapter: resp.get("ext_order_id"), resp.get("fill_price").
    """
    if engine is None or not _CAN_PG:
        return

    # 1) Orden
    try:
        notes = resp.copy() if isinstance(resp, dict) else {}
        notes["reduce_only"] = bool(reduce_only)
        insert_order(
            engine,
            strategy=strategy,
            exchange=venue,
            symbol=symbol,
            side=side,
            type_=type_,
            qty=float(qty),
            px=None,
            status=(resp.get("status", "sent") if isinstance(resp, dict) else "sent"),
            ext_order_id=(resp.get("ext_order_id") if isinstance(resp, dict) else None),
            notes=notes,
        )
    except Exception:
        log.exception("[%s] persist failure: insert_order", symbol)

    # 2) Fill
    try:
        fpx = None
        fee_usdt = None
        if isinstance(resp, dict):
            fpx = resp.get("fill_price")
            fee_usdt = resp.get("fee_usdt")
            if fee_usdt is None and resp.get("fee_bps") is not None:
                fee_usdt = abs(qty * mark_price) * float(resp["fee_bps"]) / 10000.0
        insert_fill(
            engine,
            ts=datetime.now(timezone.utc),
            venue=venue,
            strategy=strategy,
            symbol=symbol,
            side=side,
            type_=type_,
            qty=float(qty),
            price=float(fpx if fpx is not None else mark_price),
            fee_usdt=(fee_usdt if fee_usdt is not None else None),
            reduce_only=bool(reduce_only),
            ext_order_id=(resp.get("ext_order_id") if isinstance(resp, dict) else None),
            raw=resp,
        )
    except Exception:
        log.exception("[%s] persist failure: insert_fill", symbol)

    # 3) Upsert posición
    try:
        upsert_position(
            engine, venue=venue, symbol=symbol,
            qty=pos_obj.qty, avg_price=pos_obj.avg_price,
            realized_pnl=pos_obj.realized_pnl, fees_paid=pos_obj.fees_paid
        )
    except Exception:
        log.exception("[%s] persist failure: upsert_position", symbol)

    # 4) Snapshot de PnL
    try:
        px = float(pnl_mark_px if pnl_mark_px is not None else mark_price)
        insert_pnl_snapshot(
            engine, venue=venue, symbol=symbol, qty=pos_obj.qty,
            price=px, avg_price=pos_obj.avg_price,
            upnl=None,  # deja que el consumidor calcule o ajusta si tienes función
            rpnl=pos_obj.realized_pnl, fees=pos_obj.fees_paid
        )
    except Exception:
        log.exception("[%s] persist failure: insert_pnl_snapshot", symbol)

    # 5) Risk event opcional
    if risk_event_kind:
        try:
            insert_risk_event(
                engine, venue=venue, symbol=symbol,
                kind=risk_event_kind, message=risk_event_message or "",
                details=risk_event_details or {}
            )
        except Exception:
            log.exception("[%s] persist failure: insert_risk_event (%s)", symbol, risk_event_kind)
