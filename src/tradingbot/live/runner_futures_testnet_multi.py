# src/tradingbot/live/runner_futures_testnet_multi.py
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd

from .runner import BarAggregator
from ..adapters.binance_futures_ws import BinanceFuturesWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.balance import fetch_futures_usdt_free, cap_qty_by_balance_futures
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.pnl import Position, position_from_db, apply_fill, mark_to_market
from ..risk.oco import OcoBook, OcoOrder
from ..execution.normalize import adjust_order
from .common_exec import persist_bar_and_snapshot, persist_after_order

try:
    from ..storage.timescale import (
        get_engine, insert_risk_event, load_positions, load_latest_marks,
        load_active_oco_by_symbols, upsert_oco, update_oco_active, cancel_oco, mark_oco_triggered
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
    soft_cap_grace_sec: int = 30,
    sl_pct: float = 0.01,        # NEW
    tp_pct: float = 0.02,        # NEW
    cooldown_secs: int = 60,      # NEW (post SL)
    trail_pct: float | None = 0.004,
    trading_window_utc: str = "00:00-23:59",  # ventana operativa (UTC)
    daily_max_loss_usdt: float = 100.0,
    daily_max_drawdown_pct: float = 0.05,
    max_consecutive_losses: int = 3,

):
    """
    Futures UM TESTNET multi-símbolo:
      - WS multi-stream -> barras 1m
      - BreakoutATR + Risk por símbolo
      - Hard/Soft caps + auto-close reduce-only
      - Rehidratación de posiciones + PnL
      - Stop-Loss / Take-Profit (OCO simulado) + cooldown post-stop
      - Inserción de FILLs en BD
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
                # ventana cruza medianoche: activo si (hhmm >= start) o (hhmm <= end)
                ok = (hhmm >= start) or (hhmm <= end)
            return ok and (not dguard.halted)
        except Exception:
            return not dguard.halted

    def _approx_equity_total() -> float:
        total = 0.0
        for s, pos in pos_book.items():
            px = guard.st.prices.get(s) or 0.0
            total += pos.realized_pnl + abs(pos.qty) * px
        return total

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
            log.info("OCOs rehidratados (futures): %s", list(preload_items.keys()))
        except Exception as e:
            log.warning("No se pudieron rehidratar OCOs (futures): %s", e)

    # Rehidratación
    pos_book: dict[str, Position] = {}
    if engine is not None:
        try:
            dbpos = load_positions(engine, venue=venue)
            dbmk  = load_latest_marks(engine, venue=venue)
            for s in symbols:
                pos_book[s] = position_from_db(dbpos.get(s))
                guard.st.positions[s] = pos_book[s].qty
                if s in risks:
                    risks[s].set_position(pos_book[s].qty)
                if s in dbmk:
                    guard.st.prices[s] = dbmk[s]
            log.info("Rehidratadas posiciones futures: %s", {k: vars(v) for k,v in pos_book.items()})
        except Exception as e:
            log.warning("No se pudieron rehidratar posiciones futures: %s", e)
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
    taker_fee_bps = getattr(exec_adapter, "taker_fee_bps", 5.0)
    async def close_position_for(sym: str, reason_kind: str, now_price: float, qty_override: float | None = None):
        """Cierra posición futures con reduce-only en el lado opuesto a la posición."""
        q = pos_book[sym].qty
        if abs(q) <= 0:
            return False
        close_side = "sell" if q > 0 else "buy"
        qty_to_close = float(qty_override) if qty_override is not None else abs(q)
        try:
            rules = exec_adapter.meta.rules[sym]
            ok, adj_qty, adj_price, reason = adjust_order(close_side, qty_to_close, now_price, rules)
            if not ok:
                log.warning("[%s] FUTURES abort AUTO_CLOSE por filtros (%s): qty=%.10f px=%.10f", sym, reason,
                            qty_to_close, now_price)
                return False

            resp = await exec_adapter.place_order(sym, close_side, "market", adj_qty, reduce_only=True,
                                                  mark_price=adj_price)
            # PnL/posiciones
            pos_book[sym] = apply_fill(pos_book[sym], close_side, float(adj_qty), float(adj_price), taker_fee_bps,
                                       venue_type="futures")

            guard.st.positions[sym] = pos_book[sym].qty
            # --- Actualiza pérdidas realizadas (delta) ---
            delta_r = pos_book[sym].realized_pnl - last_rpnl[sym]
            if abs(delta_r) > 0:
                dguard.on_realized_delta(delta_r)
                last_rpnl[sym] = pos_book[sym].realized_pnl

            log.warning("[%s] %s ejecutado: side=%s qty=%.6f px=%.4f", sym, reason_kind, close_side, adj_qty, adj_price)

            if engine is not None:
                try:
                    persist_after_order(
                        engine,
                        venue=venue,
                        strategy="breakout_atr_futures",
                        symbol=sym,
                        side=close_side,
                        type_="market",
                        qty=float(adj_qty),
                        mark_price=float(adj_price),
                        resp=resp,
                        reduce_only=True,
                        pos_obj=pos_book[sym],
                        pnl_mark_px=float(adj_price),
                        risk_event_kind=reason_kind,
                        risk_event_message=f"{reason_kind} reduce-only close",
                        risk_event_details={"side": close_side, "qty": float(adj_qty), "price": float(adj_price)}
                    )
                except Exception:
                    log.exception("FUTURES persist failure en close_position_for (helper)")

            return True
        except Exception as e:
            log.error("[%s] cierre por %s falló: %s", sym, reason_kind, e)
            return False

    async def check_sl_tp_and_maybe_close(sym: str, last_price: float):
        """(Mantenida por compatibilidad, ya no se usa si OCO está activo)."""
        q = pos_book[sym].qty
        if abs(q) <= 0 or pos_book[sym].avg_price <= 0:
            return
        entry = pos_book[sym].avg_price

        if q > 0:
            # LONG: SL si cae; TP si sube
            sl_trigger = entry * (1.0 - sl_pct)
            tp_trigger = entry * (1.0 + tp_pct)
            if last_price <= sl_trigger:
                ok = await close_position_for(sym, "STOP_LOSS", last_price)
                if ok:
                    cooldown_until[sym] = datetime.now(timezone.utc) + timedelta(seconds=cooldown_secs)
                return
            if last_price >= tp_trigger:
                await close_position_for(sym, "TAKE_PROFIT", last_price)
        else:
            # SHORT: SL si sube; TP si cae
            sl_trigger = entry * (1.0 + sl_pct)
            tp_trigger = entry * (1.0 - tp_pct)
            if last_price >= sl_trigger:
                ok = await close_position_for(sym, "STOP_LOSS", last_price)
                if ok:
                    cooldown_until[sym] = datetime.now(timezone.utc) + timedelta(seconds=cooldown_secs)
                return
            if last_price <= tp_trigger:
                await close_position_for(sym, "TAKE_PROFIT", last_price)

    # >>> Wrapper para que OCO también active cooldown cuando sea STOP_LOSS
    async def _oco_close_fn(s, reason, qty, px):
        ok = await close_position_for(s, reason, px, qty_override=qty)
        if ok and reason == "STOP_LOSS":
            cooldown_until[s] = datetime.now(timezone.utc) + timedelta(seconds=cooldown_secs)
        return ok

    last_rpnl: dict[str, float] = {s: pos_book[s].realized_pnl for s in symbols}

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
        # --- Equity & ventana horaria / halts ---
        # equity aproximada = suma(|qty|*precio) + realized_pnl (por símbolo). Usamos tu pos_book y precio actual.
        # Para simplicidad en multi-símbolo, aproximamos con el símbolo actual:
        dguard.on_mark(datetime.now(timezone.utc), equity_now=_approx_equity_total())

        # Si fuera de ventana o halt, saltar señales (y opcionalmente cerrar todo)
        if not _in_trading_window(datetime.now(timezone.utc)):
            if dguard.halted and dguard.lim.halt_action == "close_all" and abs(pos_book[sym].qty) > 0:
                # cerrar reduce-only
                await close_position_for(sym, "HALT_CLOSE_ALL", closed.c)
            continue
        halted, reason = dguard.check_halt()
        if halted:
            log.error("[HALT] %s motivo=%s", sym, reason)
            if dguard.lim.halt_action == "close_all" and abs(pos_book[sym].qty) > 0:
                await close_position_for(sym, f"HALT_{reason}", closed.c)
            continue

        persist_bar_and_snapshot(
            engine,
            venue=venue,
            symbol=sym,
            bar=closed,
            cur_pos=guard.st.positions.get(sym, 0.0)
        )

        # === OCO simulado: evalúa SL/TP linkeados a la posición (long/short) ===
        await oco.evaluate(
            symbol=sym,
            last_price=closed.c,
            pos_qty=pos_book[sym].qty,
            entry_price=pos_book[sym].avg_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            close_fn=_oco_close_fn,
            trail_pct=trail_pct,
            tp_tiers=[(1.0, 0.5), (2.0, 0.5)]  # 50% @ 1R, 50% @ 2R
        )

        # (Eliminado) Doble chequeo manual de SL/TP para no duplicar:
        # await check_sl_tp_and_maybe_close(sym, closed.c)

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

        if not risks[sym].check_limits(closed.c):
            log.warning("[%s] RiskManager disabled; kill switch activated", sym)
            continue

        # ----- SOFT CAPS -----
        action, reason, metrics = guard.soft_cap_decision(sym, side, abs(trade_qty), closed.c)
        if action == "block":
            log.info("[%s] Orden bloqueada: %s | metrics=%s", sym, reason, metrics)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue=venue, symbol=sym, kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    log.exception("FUTURES persist failure en CAP_SOFT riskevent")

            # AUTO‑CLOSE (futures): reduce-only opuesto a la posición
            if auto_close:
                need_qty, msg, met = guard.compute_auto_close_qty(sym, closed.c)
                if need_qty > 0:
                    try:
                        cur_pos = guard.st.positions.get(sym, 0.0)
                        close_side = "sell" if cur_pos > 0 else "buy"
                        rules = exec_adapter.meta.rules[sym]
                        ok, adj_qty, adj_price, reason = adjust_order(close_side, need_qty, closed.c, rules)
                        if not ok:
                            log.warning("[%s] FUTURES abort AUTO_CLOSE por filtros (%s): qty=%.10f px=%.10f", sym,
                                        reason, need_qty, closed.c)
                            continue

                        resp = await exec_adapter.place_order(sym, close_side, "market", adj_qty, reduce_only=True,
                                                              mark_price=adj_price)
                        # PnL/posiciones
                        pos_book[sym] = apply_fill(pos_book[sym], close_side, float(adj_qty), float(adj_price),
                                                   taker_fee_bps, venue_type="futures")

                        guard.st.positions[sym] = pos_book[sym].qty
                        if sym in risks:
                            risks[sym].set_position(pos_book[sym].qty)
                        # --- Actualiza pérdidas realizadas (delta) ---
                        delta_r = pos_book[sym].realized_pnl - last_rpnl[sym]
                        if abs(delta_r) > 0:
                            dguard.on_realized_delta(delta_r)
                            last_rpnl[sym] = pos_book[sym].realized_pnl

                        log.warning("[%s] AUTO_CLOSE futures ejecutado: %s | resp=%s", sym, msg, resp)
                        if engine is not None:
                            try:
                                persist_after_order(
                                    engine,
                                    venue=venue,
                                    strategy="breakout_atr_futures",
                                    symbol=sym,
                                    side=close_side,
                                    type_="market",
                                    qty=float(adj_qty),
                                    mark_price=float(adj_price),
                                    resp=resp,
                                    reduce_only=True,
                                    pos_obj=pos_book[sym],
                                    pnl_mark_px=float(adj_price),
                                    risk_event_kind="AUTO_CLOSE",
                                    risk_event_message=msg,
                                    risk_event_details={"executed_side": close_side, "executed_qty": float(adj_qty)}
                                )
                            except Exception:
                                log.exception("FUTURES persist failure en AUTO_CLOSE (helper)")

                    except Exception as e:
                        log.error("[%s] AUTO_CLOSE futures falló: %s", sym, e)
            continue

        elif action == "soft_allow":
            if engine is not None:
                try:
                    insert_risk_event(engine, venue=venue, symbol=sym, kind="INFO", message=reason, details=metrics)
                except Exception:
                    log.exception("FUTURES persist failure en riskevent INFO")
            # continúa flujo normal

        # ----- Balance → cap qty -----
        try:
            bal = await fetch_futures_usdt_free(exec_adapter.rest)
            capped_qty = cap_qty_by_balance_futures(side, abs(trade_qty), closed.c, bal.free_usdt, leverage, utilization=0.5)
            if capped_qty <= 0:
                log.info("[%s] Skip order (USDT libre insuficiente). Free: %.4f", sym, bal.free_usdt)
                continue
        except Exception as e:
            log.warning("[%s] No se pudo leer balance futures: %s (continuo qty original)", sym, e)
            capped_qty = abs(trade_qty)

        # ----- Enviar orden normal -----
        try:
            rules = exec_adapter.meta.rules[sym]
            ok, adj_qty, adj_price, reason = adjust_order(side, capped_qty, closed.c, rules)
            if not ok:
                log.warning("[%s] FUTURES abort order por filtros (%s): qty=%.10f px=%.10f", sym, reason, capped_qty,
                            closed.c)
                continue

            resp = await exec_adapter.place_order(sym, side, "market", adj_qty, mark_price=adj_price)
            log.info("[%s] FUTURES TESTNET FILL %s", sym, resp)

            # PnL/posiciones (usar adj_qty)
            pos_book[sym] = apply_fill(pos_book[sym], side, float(adj_qty), float(adj_price), taker_fee_bps,
                                       venue_type="futures")
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
                    persist_after_order(
                        engine,
                        venue=venue,
                        strategy="breakout_atr_futures",
                        symbol=sym,
                        side=side,
                        type_="market",
                        qty=float(adj_qty),
                        mark_price=float(adj_price),
                        resp=resp,
                        reduce_only=False,
                        pos_obj=pos_book[sym],
                        pnl_mark_px=float(adj_price),
                        risk_event_kind=None
                    )
                except Exception:
                    log.exception("FUTURES persist failure tras orden (helper)")
        except Exception as e:
            log.error("[%s] Error orden futures testnet: %s", sym, e)
