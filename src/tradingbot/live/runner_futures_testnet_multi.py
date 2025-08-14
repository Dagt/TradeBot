# src/tradingbot/live/runner_futures_testnet_multi.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
import pandas as pd

from .runner import BarAggregator
from ..adapters.binance_futures_ws import BinanceFuturesWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.balance import fetch_futures_usdt_free, cap_qty_by_balance_futures
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.pnl import Position, position_from_db, apply_fill, mark_to_market

try:
    from ..storage.timescale import (
        get_engine, insert_order, insert_bar_1m, insert_portfolio_snapshot, insert_risk_event,
        upsert_position, insert_pnl_snapshot, load_positions, load_latest_marks
    )
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

async def run_live_binance_futures_testnet_multi(
    symbols: list[str],
    leverage: int = 5,
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 2000.0,
    per_symbol_cap_usdt: float = 1000.0,
    auto_close: bool = True,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30
):
    """
    Futures UM TESTNET multi-símbolo con soft/hard caps + auto-close reduce-only.
    Incluye rehidratación de posiciones y PnL.
    """
    venue = "binance_futures_um_testnet"
    symbols = [s.upper().replace("-", "/") for s in symbols]
    ws = BinanceFuturesWSAdapter()
    exec_adapter = BinanceFuturesAdapter(leverage=leverage, testnet=True)

    aggs = {s: BarAggregator() for s in symbols}
    strats = {s: BreakoutATR() for s in symbols}
    risks = {s: RiskManager(max_pos=1.0) for s in symbols}

    guard = PortfolioGuard(GuardConfig(
        total_cap_usdt=total_cap_usdt,
        per_symbol_cap_usdt=per_symbol_cap_usdt,
        venue=venue,
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec
    ))

    engine = get_engine() if (persist_pg and _CAN_PG) else None

    # Rehidratación
    pos_book: dict[str, Position] = {}
    if engine is not None:
        try:
            dbpos = load_positions(engine, venue=venue)
            dbmk  = load_latest_marks(engine, venue=venue)
            for s in symbols:
                pos_book[s] = position_from_db(dbpos.get(s))
                guard.st.positions[s] = pos_book[s].qty
                if s in dbmk:
                    guard.st.prices[s] = dbmk[s]
            log.info("Rehidratadas posiciones futures: %s", {k: vars(v) for k,v in pos_book.items()})
        except Exception as e:
            log.warning("No se pudieron rehidratar posiciones futures: %s", e)
            for s in symbols:
                pos_book[s] = Position()
    else:
        for s in symbols:
            pos_book[s] = Position()

    taker_fee_bps = 2.0  # típico futures

    async for t in ws.stream_trades_multi(symbols, channel="aggTrade"):
        sym = t["symbol"].upper().replace("-", "/")
        if sym not in aggs:
            continue

        ts: datetime = t["ts"] or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t["qty"] or 0.0)

        guard.mark_price(sym, px)

        closed = aggs[sym].on_trade(ts, px, qty)
        if closed is None:
            continue

        if engine is not None:
            try:
                insert_bar_1m(engine, exchange=venue, symbol=sym,
                              ts=closed.ts_open, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
            except Exception:
                pass
            try:
                cur_pos = guard.st.positions.get(sym, 0.0)
                insert_portfolio_snapshot(engine, venue=venue, symbol=sym,
                                          position=cur_pos, price=closed.c, notional_usd=abs(cur_pos)*closed.c)
            except Exception:
                pass

        df: pd.DataFrame = aggs[sym].last_n_bars_df(200)
        if len(df) < 140:
            continue

        sig = strats[sym].on_bar({"window": df})
        if sig is None:
            continue

        delta = risks[sym].size(sig.side, sig.strength)
        if abs(delta) < 1e-9:
            continue

        side = "buy" if delta > 0 else "sell"

        # ***** SOFT CAPS: decisión *****
        action, reason, metrics = guard.soft_cap_decision(sym, side, abs(trade_qty), closed.c)
        if action == "block":
            log.info("[%s] Orden bloqueada: %s | metrics=%s", sym, reason, metrics)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue=venue, symbol=sym, kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    pass

            # AUTO‑CLOSE (futures): reduce-only en el lado opuesto a la posición
            if auto_close:
                need_qty, msg, met = guard.compute_auto_close_qty(sym, closed.c)
                if need_qty > 0:
                    try:
                        cur_pos = guard.st.positions.get(sym, 0.0)
                        close_side = "sell" if cur_pos > 0 else "buy"
                        resp = await exec_adapter.place_order(sym, close_side, "market", need_qty, reduce_only=True, mark_price=closed.c)
                        # Actualiza PnL y guard
                        pos_book[sym] = apply_fill(pos_book[sym], close_side, float(need_qty), float(closed.c), taker_fee_bps, venue_type="futures")
                        guard.st.positions[sym] = pos_book[sym].qty
                        log.warning("[%s] AUTO_CLOSE futures ejecutado: %s | resp=%s", sym, msg, resp)
                        if engine is not None:
                            try:
                                upsert_position(engine, venue=venue, symbol=sym,
                                                qty=pos_book[sym].qty, avg_price=pos_book[sym].avg_price,
                                                realized_pnl=pos_book[sym].realized_pnl, fees_paid=pos_book[sym].fees_paid)
                                upnl = mark_to_market(pos_book[sym], closed.c)
                                insert_pnl_snapshot(engine, venue=venue, symbol=sym, qty=pos_book[sym].qty,
                                                    price=closed.c, avg_price=pos_book[sym].avg_price,
                                                    upnl=upnl, rpnl=pos_book[sym].realized_pnl, fees=pos_book[sym].fees_paid)
                                insert_risk_event(engine, venue=venue, symbol=sym, kind="AUTO_CLOSE", message=msg,
                                                  details={"executed_side": close_side, "executed_qty": need_qty})
                            except Exception:
                                pass
                    except Exception as e:
                        log.error("[%s] AUTO_CLOSE futures falló: %s", sym, e)
            continue
        elif action == "soft_allow":
            if engine is not None:
                try:
                    insert_risk_event(engine, venue=venue, symbol=sym, kind="INFO", message=reason, details=metrics)
                except Exception:
                    pass
            # sigue flujo normal (balances + orden)

        # Cap por balance (USDT libre * leverage)
        try:
            bal = await fetch_futures_usdt_free(exec_adapter.rest)
            capped_qty = cap_qty_by_balance_futures(side, abs(trade_qty), closed.c, bal.free_usdt, leverage, utilization=0.5)
            if capped_qty <= 0:
                log.info("[%s] Skip order (USDT libre insuficiente). Free: %.4f", sym, bal.free_usdt)
                continue
        except Exception as e:
            log.warning("[%s] No se pudo leer balance futures: %s (continuo qty original)", sym, e)
            capped_qty = abs(trade_qty)

        # Ejecuta orden y actualiza PnL/posiciones
        try:
            resp = await exec_adapter.place_order(sym, side, "market", capped_qty, mark_price=closed.c)
            log.info("[%s] FUTURES TESTNET FILL %s", sym, resp)

            pos_book[sym] = apply_fill(pos_book[sym], side, float(capped_qty), float(closed.c), taker_fee_bps, venue_type="futures")
            guard.st.positions[sym] = pos_book[sym].qty

            if engine is not None:
                try:
                    insert_order(engine,
                                 strategy="breakout_atr_futures",
                                 exchange=venue,
                                 symbol=sym,
                                 side=side,
                                 type_="market",
                                 qty=float(capped_qty),
                                 px=None,
                                 status=resp.get("status", "sent"),
                                 ext_order_id=str(resp.get("resp", {}).get("id") if isinstance(resp.get("resp"), dict) else None),
                                 notes=resp)
                    
                except Exception:
                    pass
                try:
                    upsert_position(engine, venue=venue, symbol=sym,
                                    qty=pos_book[sym].qty, avg_price=pos_book[sym].avg_price,
                                    realized_pnl=pos_book[sym].realized_pnl, fees_paid=pos_book[sym].fees_paid)
                    upnl = mark_to_market(pos_book[sym], closed.c)
                    insert_pnl_snapshot(engine, venue=venue, symbol=sym, qty=pos_book[sym].qty,
                                        price=closed.c, avg_price=pos_book[sym].avg_price,
                                        upnl=upnl, rpnl=pos_book[sym].realized_pnl, fees=pos_book[sym].fees_paid)
                except Exception:
                    pass
        except Exception as e:
            log.error("[%s] Error orden futures testnet: %s", sym, e)
