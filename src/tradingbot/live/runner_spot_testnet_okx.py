# src/tradingbot/live/runner_spot_testnet_okx.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
import pandas as pd

from ..execution.normalize import adjust_order

from .runner import BarAggregator  # el mismo agregador de 1m
from ..adapters.okx_spot import OKXSpotAdapter as OKXSpotWSAdapter
from ..adapters.okx_spot import OKXSpotAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from .common_exec import persist_bar_and_snapshot, persist_after_order


# Persistencia opcional
try:
    from ..storage.timescale import get_engine, insert_order, insert_bar_1m, insert_risk_event
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

async def run_live_okx_spot_testnet(
    symbol: str = "BTC/USDT",
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30
):
    """
    WS spot testnet -> barras 1m -> BreakoutATR -> Risk -> OKXSpotAdapter (orden real testnet)
    """
    ws = OKXSpotWSAdapter()
    exec_adapter = OKXSpotAdapter()
    agg = BarAggregator()
    strat = BreakoutATR()
    risk = RiskManager(max_pos=1.0)
    guard = PortfolioGuard(GuardConfig(
        total_cap_usdt=total_cap_usdt,
        per_symbol_cap_usdt=per_symbol_cap_usdt,
        venue="okx_spot_testnet",
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec
    ))

    # Guardia diaria / ventana (ajusta los límites a tu gusto)
    trading_window_utc = "00:00-23:59"
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_usdt=100.0,  # pérdida diaria máxima
        daily_max_drawdown_pct=0.05,  # dd intradía, 5%
        max_consecutive_losses=3,  # racha de pérdidas
        halt_action="close_all"  # en spot sólo bloquearemos señales (no hay pos abiertas en este runner)
    ), venue="okx_spot_testnet")

    def _in_trading_window(now_utc: datetime) -> bool:
        try:
            hhmm = now_utc.strftime("%H:%M")
            start, end = trading_window_utc.split("-")
            if start <= end:
                ok = (start <= hhmm <= end)
            else:
                ok = (hhmm >= start) or (hhmm <= end)
            return ok and (not dguard.halted)
        except Exception:
            return not dguard.halted

    engine = get_engine() if (persist_pg and _CAN_PG) else None

    async for t in ws.stream_trades(symbol):
        ts: datetime = t["ts"] or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t["qty"] or 0.0)

        closed = agg.on_trade(ts, px, qty)
        if closed is None:
            continue

        guard.mark_price(symbol, px)

        # Marca equity aproximada y verifica ventana / halts
        # (en spot sin posiciones, la equity real no aplica; marcamos cero)
        dguard.on_mark(datetime.now(timezone.utc), equity_now=0.0)

        if not _in_trading_window(datetime.now(timezone.utc)):
            # fuera de ventana o halt -> no generamos señales
            continue

        halted, reason = dguard.check_halt()
        if halted:
            log.error("[HALT SPOT] motivo=%s", reason)
            # (opcional) persiste el evento
            if engine is not None:
                try:
                    insert_risk_event(engine, venue="okx_spot_testnet", symbol=symbol,
                                      kind=f"HALT_{reason}", message=f"SPOT halt: {reason}", details={})
                except Exception:
                    log.exception("SPOT persist failure: risk_event HALT")
            break

        # Persistir barra 1m (opcional)
        persist_bar_and_snapshot(engine, venue="okx_spot_testnet", symbol=symbol, bar=closed, cur_pos=None)

        df: pd.DataFrame = agg.last_n_bars_df(200)
        if len(df) < 140:
            continue

        sig = strat.on_bar({"window": df})
        if sig is None:
            continue

        delta = risk.size(sig.side, sig.strength)
        if abs(delta) < 1e-9:
            continue

        side = "buy" if delta > 0 else "sell"

        if not risk.check_limits(closed.c):
            log.warning("RiskManager disabled; kill switch activated")
            continue

        # --- Portfolio guard ---
        action, reason, metrics = guard.soft_cap_decision(symbol, side, abs(trade_qty), closed.c)
        if action == "block":
            log.warning("[PG] Bloqueado %s: %s", symbol, reason)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue="okx_spot_testnet", symbol=symbol,
                                      kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    log.exception("SPOT persist failure: risk_event VIOLATION")
            continue
        elif action == "soft_allow" and engine is not None:
            try:
                insert_risk_event(engine, venue="okx_spot_testnet", symbol=symbol,
                                  kind="INFO", message=reason, details=metrics)
            except Exception:
                log.exception("SPOT persist failure: risk_event INFO")

        # --- NEW: cap por balance spot ---
        from ..execution.balance import fetch_spot_balances, cap_qty_by_balance_spot
        try:
            base, quote = symbol.split("/")
            balances = await fetch_spot_balances(exec_adapter.rest, base, quote)
            capped_qty = cap_qty_by_balance_spot(side, abs(trade_qty), closed.c, balances, utilization=0.95)
            if capped_qty <= 0:
                log.info("Skip order (saldo insuficiente). Balances: %s", balances)
                continue
        except Exception as e:
            log.warning("No se pudo leer balances spot: %s (continuo con qty original)", e)
            capped_qty = abs(trade_qty)

        # --- Envío de la orden (API) ---
        try:
            rules = exec_adapter.meta.rules[symbol]  # debe existir en tu adapter/meta
            ok, adj_qty, adj_price, reason = adjust_order(side, capped_qty, closed.c, rules)
            if not ok:
                log.warning(
                    "SPOT abort order por filtros (%s): symbol=%s qty=%.10f price=%.10f",
                    reason, symbol, capped_qty, closed.c
                )
                continue

            prev_rpnl = 0.0
            resp = await exec_adapter.place_order(symbol, side, "market", adj_qty, mark_price=adj_price)
            log.info("SPOT TESTNET FILL %s", resp)

            # Actualiza RiskManager con la cantidad realmente enviada
            try:
                risk.add_fill(side, adj_qty)
                guard.update_position_on_order(symbol, side, float(adj_qty))
                delta_rpnl = resp.get("realized_pnl", 0.0) - prev_rpnl
                dguard.on_realized_delta(delta_rpnl)
                dguard.on_mark(datetime.now(timezone.utc), equity_now=0.0)
                halted, reason = dguard.check_halt()
                if halted:
                    log.error("[HALT SPOT] motivo=%s", reason)
                    if engine is not None:
                        try:
                            insert_risk_event(engine, venue="okx_spot_testnet", symbol=symbol,
                                              kind=f"HALT_{reason}", message=f"SPOT halt: {reason}", details={})
                        except Exception:
                            log.exception("SPOT persist failure: risk_event HALT")
                    break
                try:
                    if engine is not None:
                        insert_risk_event(engine, venue="okx_spot_testnet", symbol=symbol,
                                          kind="INFO", message="ORDER_SENT",
                                          details={"qty": float(adj_qty), "side": side})
                except Exception:
                    log.exception("SPOT persist failure: risk_event ORDER_SENT")

            except Exception:
                pass

        except Exception:
            log.exception("SPOT fallo al ENVIAR orden al exchange (place_order)")
            continue  # seguimos con el siguiente tick/iteración del loop

        # --- Persistencia (BD) ---
        persist_after_order(
            engine,
            venue="okx_spot_testnet",
            strategy="breakout_atr_spot",
            symbol=symbol,
            side=side,
            type_="market",
            qty=float(adj_qty),
            mark_price=float(adj_price),
            resp=resp,
            reduce_only=False,
            pos_obj=risk,
            pnl_mark_px=float(adj_price),
            risk_event_kind=None
        )

