# src/tradingbot/live/runner_spot_testnet_multi.py
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd

from .runner import BarAggregator
from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.balance import fetch_spot_balances, cap_qty_by_balance_spot
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.pnl import Position, position_from_db, apply_fill, mark_to_market
from ..risk.oco import OcoBook, OcoOrder

# Persistencia opcional
try:
    from ..storage.timescale import (
        get_engine, insert_order, insert_bar_1m, insert_portfolio_snapshot, insert_risk_event,
        upsert_position, insert_pnl_snapshot, load_positions, load_latest_marks, insert_fill,
        load_active_oco_by_symbols, upsert_oco, update_oco_active, cancel_oco, mark_oco_triggered
    )
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

async def run_live_binance_spot_testnet_multi(
    symbols: list[str],
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    auto_close: bool = True,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    sl_pct: float = 0.01,
    tp_pct: float = 0.02,
    cooldown_secs: int = 60,
    trail_pct: float | None = 0.004,
    trading_window_utc: str = "00:00-23:59",
    daily_max_loss_usdt: float = 50.0,
    daily_max_drawdown_pct: float = 0.05,
    max_consecutive_losses: int = 3,
):
    """
    Spot TESTNET multi-símbolo:
      - WS multi-stream -> barras 1m
      - BreakoutATR + Risk por símbolo
      - Hard/Soft caps + auto-close
      - Rehidratación de posiciones + PnL
      - OCO simulado (SL/TP) + cooldown post-SL
      - Inserción de FILLs en BD
    """
    venue = "binance_spot_testnet"
    symbols = [s.upper().replace("-", "/") for s in symbols]
    ws = BinanceSpotWSAdapter()
    exec_adapter = BinanceSpotAdapter()

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

    # === OCO con persistencia (callbacks) ===
    def _on_create(symbol: str, order: OcoOrder):
        if persist_pg and engine is not None:
            upsert_oco(engine, venue=venue, symbol=symbol, side=order.side,
                       qty=order.qty, entry_price=order.entry_price,
                       sl_price=order.sl_price, tp_price=order.tp_price, status="active")

    def _on_update(symbol: str, order: OcoOrder):
        if persist_pg and engine is not None:
            update_oco_active(engine, venue=venue, symbol=symbol,
                              qty=order.qty, entry_price=order.entry_price,
                              sl_price=order.sl_price, tp_price=order.tp_price)

    def _on_cancel(symbol: str):
        if persist_pg and engine is not None:
            cancel_oco(engine, venue=venue, symbol=symbol)

    def _on_trigger(symbol: str, which: str):
        if persist_pg and engine is not None:
            mark_oco_triggered(engine, venue=venue, symbol=symbol, triggered=which)

    oco = OcoBook(
        on_create=_on_create,
        on_update=_on_update,
        on_cancel=_on_cancel,
        on_trigger=_on_trigger
    )

    # === Guardia diaria (pérdidas/drawdown/racha) ===
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        halt_action="close_all"
    ), venue=venue)

    def _in_trading_window(now_utc: datetime) -> bool:
        try:
            hhmm = now_utc.strftime("%H:%M")
            start, end = trading_window_utc.split("-")
            if start <= end:
                ok = (start <= hhmm <= end)
            else:
                # Ventana cruza medianoche: activo si (hhmm >= start) o (hhmm <= end)
                ok = (hhmm >= start) or (hhmm <= end)
            return ok and (not dguard.halted)
        except Exception:
            return not dguard.halted

    # === Rehidratar OCOs activos desde BD (si existen) ===
    if persist_pg and engine is not None:
        try:
            persisted = load_active_oco_by_symbols(engine, venue=venue, symbols=symbols)
            preload_items = {}
            for sym, r in persisted.items():
                preload_items[sym] = OcoOrder(
                    symbol=sym, side=r["side"], qty=r["qty"],
                    entry_price=r["entry_price"], sl_price=r["sl_price"], tp_price=r["tp_price"]
                )
            oco.preload(preload_items)
            log.info("OCOs rehidratados (spot): %s", list(preload_items.keys()))
        except Exception as e:
            log.warning("No se pudieron rehidratar OCOs (spot): %s", e)

    # Rehidratación
    pos_book: dict[str, Position] = {}
    def _approx_equity_total() -> float:
        total = 0.0
        for s, pos in pos_book.items():
            px = guard.st.prices.get(s) or 0.0
            total += pos.realized_pnl + max(0.0, pos.qty) * px  # spot: solo long
        return total

    if engine is not None:
        try:
            dbpos = load_positions(engine, venue=venue)
            dbmk  = load_latest_marks(engine, venue=venue)
            for s in symbols:
                pos_book[s] = position_from_db(dbpos.get(s))
                guard.st.positions[s] = pos_book[s].qty
                if s in dbmk:
                    guard.st.prices[s] = dbmk[s]
            log.info("Rehidratadas posiciones spot: %s", {k: vars(v) for k,v in pos_book.items()})
            # --- NUEVO: sincroniza RiskManager con qty rehidratado ---
            for s in symbols:
                if s in risks:
                    risks[s].set_position(pos_book[s].qty)

        except Exception as e:
            log.warning("No se pudieron rehidratar posiciones spot: %s", e)
            for s in symbols:
                pos_book[s] = Position()
                guard.st.positions[s] = 0.0
                if s in risks:
                    risks[s].set_position(0.0)

    else:
        for s in symbols:
            pos_book[s] = Position()
            guard.st.positions[s] = 0.0
            if s in risks:
                risks[s].set_position(0.0)


    cooldown_until: dict[str, datetime] = {s: datetime.fromtimestamp(0, tz=timezone.utc) for s in symbols}
    taker_fee_bps = getattr(exec_adapter, "taker_fee_bps", 10.0)
    async def close_position_for(sym: str, reason_kind: str, now_price: float, qty_override: float | None = None):
        """Cierra posición spot (long) con market SELL y persiste todo."""
        qty_to_close = float(qty_override) if qty_override is not None else max(0.0, pos_book[sym].qty)
        if qty_to_close <= 0:
            return False
        try:
            resp = await exec_adapter.place_order(sym, "sell", "market", qty_to_close, mark_price=now_price)
            # PnL/posiciones
            pos_book[sym] = apply_fill(pos_book[sym], "sell", float(qty_to_close), float(now_price), taker_fee_bps, venue_type="spot")
            guard.st.positions[sym] = pos_book[sym].qty
            # actualizar RiskManager tras fill ---
            if sym in risks:
                risks[sym].set_position(pos_book[sym].qty)

            log.warning("[%s] %s ejecutado: qty=%.6f px=%.4f", sym, reason_kind, qty_to_close, now_price)
            # --- Actualiza pérdidas realizadas (delta) ---
            delta_r = pos_book[sym].realized_pnl - last_rpnl[sym]
            if abs(delta_r) > 0:
                dguard.on_realized_delta(delta_r)
                last_rpnl[sym] = pos_book[sym].realized_pnl

            if engine is not None:
                try:
                    # Orden + Fill
                    insert_order(engine,
                                 strategy="breakout_atr_spot",
                                 exchange=venue,
                                 symbol=sym,
                                 side="sell",
                                 type_="market",
                                 qty=float(qty_to_close),
                                 px=None,
                                 status=resp.get("status", "sent"),
                                 ext_order_id=str(resp.get("ext_order_id") or None),
                                 notes=resp)
                    insert_fill(engine,
                        ts=datetime.now(timezone.utc), venue=venue, strategy="breakout_atr_spot",
                        symbol=sym, side="sell", type_="market",
                        qty=float(qty_to_close),
                        price=float(resp.get("fill_price") or now_price),
                        fee_usdt=(resp.get("fee_usdt") if resp.get("fee_usdt") is not None else None),
                        reduce_only=False, ext_order_id=str(resp.get("ext_order_id") or None),
                        raw=resp
                    )
                    # PnL snapshots + evento
                    upsert_position(engine, venue=venue, symbol=sym,
                                    qty=pos_book[sym].qty, avg_price=pos_book[sym].avg_price,
                                    realized_pnl=pos_book[sym].realized_pnl, fees_paid=pos_book[sym].fees_paid)
                    upnl = mark_to_market(pos_book[sym], now_price)
                    insert_pnl_snapshot(engine, venue=venue, symbol=sym, qty=pos_book[sym].qty,
                                        price=now_price, avg_price=pos_book[sym].avg_price,
                                        upnl=upnl, rpnl=pos_book[sym].realized_pnl, fees=pos_book[sym].fees_paid)
                    insert_risk_event(engine, venue=venue, symbol=sym, kind=reason_kind, message=f"{reason_kind} market close", details={"qty": qty_to_close, "price": now_price})
                except Exception:
                    log.exception("Error persistiendo orden/fill/PNL/riskevent (spot)")
            return True
        except Exception as e:
            log.error("[%s] cierre por %s falló: %s", sym, reason_kind, e)
            return False

    # Wrapper para que OCO también active cooldown cuando sea STOP_LOSS
    async def _oco_close_fn(s, reason, qty, px):
        ok = await close_position_for(s, reason, px, qty_override=qty)
        if ok and reason == "STOP_LOSS":
            cooldown_until[s] = datetime.now(timezone.utc) + timedelta(seconds=cooldown_secs)
        return ok

    last_rpnl: dict[str, float] = {s: pos_book[s].realized_pnl for s in symbols}

    async for t in ws.stream_trades_multi(symbols):
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

        # --- Equity & ventana horaria / halts ---
        dguard.on_mark(datetime.now(timezone.utc), equity_now=_approx_equity_total())

        if not _in_trading_window(datetime.now(timezone.utc)):
            if dguard.halted and dguard.lim.halt_action == "close_all" and pos_book[sym].qty > 0:
                await close_position_for(sym, "HALT_CLOSE_ALL", closed.c, qty_override=pos_book[sym].qty)
            continue

        halted, reason = dguard.check_halt()
        if halted:
            log.error("[HALT] %s motivo=%s", sym, reason)
            if dguard.lim.halt_action == "close_all" and pos_book[sym].qty > 0:
                await close_position_for(sym, f"HALT_{reason}", closed.c, qty_override=pos_book[sym].qty)
            continue

        # Persistencia de barra + snapshot de portafolio

        # Persistencia de barra + snapshot de portafolio
        if engine is not None:
            try:
                insert_bar_1m(engine, exchange=venue, symbol=sym,
                              ts=closed.ts_open, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
            except Exception:
                log.exception("Error insertando barra/snapshot (spot)")
            try:
                cur_pos = guard.st.positions.get(sym, 0.0)
                insert_portfolio_snapshot(engine, venue=venue, symbol=sym,
                                          position=cur_pos, price=closed.c, notional_usd=abs(cur_pos)*closed.c)
            except Exception:
                log.exception("Error insertando barra/snapshot (spot)")

        # === OCO simulado: evalúa SL/TP linkeados a la posición ===
        await oco.evaluate(
            symbol=sym,
            last_price=closed.c,
            pos_qty=pos_book[sym].qty,
            entry_price=pos_book[sym].avg_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            close_fn=_oco_close_fn,
            trail_pct=trail_pct,
            tp_tiers=[(1.0, 0.5), (2.0, 0.5)],  # 50% @ 1R, 50% @ 2R

        )

        # Cooldown post-stop (solo tras SL)
        if datetime.now(timezone.utc) < cooldown_until[sym]:
            remain = (cooldown_until[sym] - datetime.now(timezone.utc)).total_seconds()
            log.debug("[%s] EN COOLDOWN (%.1fs restantes). Omitiendo señales.", sym, remain)
            continue

        # Señal de estrategia
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

        # ----- SOFT CAPS -----
        action, reason, metrics = guard.soft_cap_decision(sym, side, abs(trade_qty), closed.c)
        if action == "block":
            log.info("[%s] Orden bloqueada: %s | metrics=%s", sym, reason, metrics)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue=venue, symbol=sym, kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    log.exception("Error persistiendo AUTO_CLOSE (spot)")

            # AUTO‑CLOSE (spot): vende para reducir si hay posición > 0
            if auto_close:
                need_qty, msg, met = guard.compute_auto_close_qty(sym, closed.c)
                if need_qty > 0 and guard.st.positions.get(sym, 0.0) > 0:
                    try:
                        capped_qty_close = min(need_qty, abs(guard.st.positions.get(sym, 0.0)))
                        if capped_qty_close > 0:
                            resp = await exec_adapter.place_order(sym, "sell", "market", capped_qty_close, mark_price=closed.c)
                            # PnL/posiciones
                            pos_book[sym] = apply_fill(pos_book[sym], "sell", float(capped_qty_close), float(closed.c), taker_fee_bps, venue_type="spot")
                            guard.st.positions[sym] = pos_book[sym].qty
                            if sym in risks:
                                risks[sym].set_position(pos_book[sym].qty)
                            # --- Actualiza pérdidas realizadas (delta) ---
                            delta_r = pos_book[sym].realized_pnl - last_rpnl[sym]
                            if abs(delta_r) > 0:
                                dguard.on_realized_delta(delta_r)
                                last_rpnl[sym] = pos_book[sym].realized_pnl

                            log.warning("[%s] AUTO_CLOSE spot ejecutado: %s | resp=%s", sym, msg, resp)
                            if engine is not None:
                                try:
                                    insert_order(engine,
                                                 strategy="breakout_atr_spot",
                                                 exchange=venue,
                                                 symbol=sym,
                                                 side="sell",
                                                 type_="market",
                                                 qty=float(capped_qty_close),
                                                 px=None,
                                                 status=resp.get("status", "sent"),
                                                 ext_order_id=str(resp.get("ext_order_id") or None),
                                                 notes=resp)
                                    insert_fill(engine,
                                        ts=datetime.now(timezone.utc), venue=venue, strategy="breakout_atr_spot",
                                        symbol=sym, side="sell", type_="market",
                                        qty=float(capped_qty_close),
                                        price=float(resp.get("fill_price") or closed.c),
                                        fee_usdt=(resp.get("fee_usdt") if resp.get("fee_usdt") is not None else None),
                                        reduce_only=False, ext_order_id=str(resp.get("ext_order_id") or None),
                                        raw=resp
                                    )
                                    upsert_position(engine, venue=venue, symbol=sym,
                                                    qty=pos_book[sym].qty, avg_price=pos_book[sym].avg_price,
                                                    realized_pnl=pos_book[sym].realized_pnl, fees_paid=pos_book[sym].fees_paid)
                                    upnl = mark_to_market(pos_book[sym], closed.c)
                                    insert_pnl_snapshot(engine, venue=venue, symbol=sym, qty=pos_book[sym].qty,
                                                        price=closed.c, avg_price=pos_book[sym].avg_price,
                                                        upnl=upnl, rpnl=pos_book[sym].realized_pnl, fees=pos_book[sym].fees_paid)
                                    insert_risk_event(engine, venue=venue, symbol=sym, kind="AUTO_CLOSE", message=msg,
                                                      details={"executed_qty": capped_qty_close})
                                except Exception:
                                    log.exception("Error persistiendo orden/fill/PNL/riskevent (spot)")
                    except Exception as e:
                        log.error("[%s] AUTO_CLOSE spot falló: %s", sym, e)
            continue

        elif action == "soft_allow":
            if engine is not None:
                try:
                    insert_risk_event(engine, venue=venue, symbol=sym, kind="INFO", message=reason, details=metrics)
                except Exception:
                    log.exception("Error persistiendo orden/fill/PNL/riskevent (spot)")
            # continúa flujo normal

        # ----- Balances → cap qty -----
        try:
            base, quote = sym.split("/")
            balances = await fetch_spot_balances(exec_adapter.rest, base, quote)
            capped_qty = cap_qty_by_balance_spot(side, abs(trade_qty), closed.c, balances, utilization=0.95)
            if capped_qty <= 0:
                log.info("[%s] Skip order (saldo insuficiente). Balances: %s", sym, balances)
                continue
        except Exception as e:
            log.warning("[%s] No se pudo leer balances spot: %s (continuo qty original)", sym, e)
            capped_qty = abs(trade_qty)

        # ----- Enviar orden normal -----
        try:
            resp = await exec_adapter.place_order(sym, side, "market", capped_qty, mark_price=closed.c)
            log.info("[%s] SPOT TESTNET FILL %s", sym, resp)

            # PnL/posiciones
            pos_book[sym] = apply_fill(pos_book[sym], side, float(capped_qty), float(closed.c), taker_fee_bps, venue_type="spot")
            guard.st.positions[sym] = pos_book[sym].qty
            if sym in risks:
                risks[sym].set_position(pos_book[sym].qty)

            # --- Actualiza pérdidas realizadas (delta) ---
            delta_r = pos_book[sym].realized_pnl - last_rpnl[sym]
            if abs(delta_r) > 0:
                dguard.on_realized_delta(delta_r)
                last_rpnl[sym] = pos_book[sym].realized_pnl

            if engine is not None:
                try:
                    insert_order(engine,
                                 strategy="breakout_atr_spot",
                                 exchange=venue,
                                 symbol=sym,
                                 side=side,
                                 type_="market",
                                 qty=float(capped_qty),
                                 px=None,
                                 status=resp.get("status", "sent"),
                                 ext_order_id=str(resp.get("ext_order_id") or None),
                                 notes=resp)
                    insert_fill(engine,
                        ts=datetime.now(timezone.utc), venue=venue, strategy="breakout_atr_spot",
                        symbol=sym, side=side, type_="market",
                        qty=float(capped_qty),
                        price=float(resp.get("fill_price") or closed.c),
                        fee_usdt=(resp.get("fee_usdt") if resp.get("fee_usdt") is not None else None),
                        reduce_only=False, ext_order_id=str(resp.get("ext_order_id") or None),
                        raw=resp
                    )
                except Exception:
                    log.exception("Error persistiendo fill/PNL (spot)")
                try:
                    upsert_position(engine, venue=venue, symbol=sym,
                                    qty=pos_book[sym].qty, avg_price=pos_book[sym].avg_price,
                                    realized_pnl=pos_book[sym].realized_pnl, fees_paid=pos_book[sym].fees_paid)
                    upnl = mark_to_market(pos_book[sym], closed.c)
                    insert_pnl_snapshot(engine, venue=venue, symbol=sym, qty=pos_book[sym].qty,
                                        price=closed.c, avg_price=pos_book[sym].avg_price,
                                        upnl=upnl, rpnl=pos_book[sym].realized_pnl, fees=pos_book[sym].fees_paid)
                except Exception:
                    log.exception("Error persistiendo fill/PNL (spot)")
        except Exception as e:
            log.error("[%s] Error orden spot testnet: %s", sym, e)
