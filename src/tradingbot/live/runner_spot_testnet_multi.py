# src/tradingbot/live/runner_spot_testnet_multi.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
import pandas as pd

from .runner import BarAggregator
from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.balance import fetch_spot_balances, cap_qty_by_balance_spot
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.pnl import Position, apply_fill, mark_to_market
from ..storage.timescale import upsert_position, insert_pnl_snapshot

# Persistencia opcional
try:
    from ..storage.timescale import get_engine, insert_order, insert_bar_1m, insert_portfolio_snapshot, insert_risk_event
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
    auto_close: bool = True   # NEW
):
    symbols = [s.upper().replace("-", "/") for s in symbols]
    ws = BinanceSpotWSAdapter()
    exec_adapter = BinanceSpotAdapter()

    aggs = {s: BarAggregator() for s in symbols}
    strats = {s: BreakoutATR() for s in symbols}
    risks = {s: RiskManager(max_pos=1.0) for s in symbols}

    guard = PortfolioGuard(GuardConfig(total_cap_usdt=total_cap_usdt,
                                       per_symbol_cap_usdt=per_symbol_cap_usdt,
                                       venue="binance_spot_testnet"))

    engine = get_engine() if (persist_pg and _CAN_PG) else None

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

        if engine is not None:
            try:
                insert_bar_1m(engine, exchange="binance_spot_testnet", symbol=sym,
                              ts=closed.ts_open, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
            except Exception as e:
                log.debug("No se pudo insertar barra 1m (%s): %s", sym, e)
            try:
                pos = guard.st.positions.get(sym, 0.0)
                notional = abs(pos) * closed.c
                insert_portfolio_snapshot(engine, venue="binance_spot_testnet",
                                          symbol=sym, position=pos, price=closed.c, notional_usd=notional)
            except Exception as e:
                log.debug("No se pudo insertar snapshot portafolio: %s", e)

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

        # Chequeo de límites previos
        exceed, reason, metrics = guard.would_exceed_caps(sym, side, abs(trade_qty), closed.c)
        if exceed:
            log.info("[%s] Orden bloqueada por guardia: %s | metrics=%s", sym, reason, metrics)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue="binance_spot_testnet", symbol=sym,
                                      kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    pass

            # AUTO‑CLOSE (SPOT): sólo puede reducir posiciones largas (vender). Si pos<=0, no hace nada.
            if auto_close:
                need_qty, msg, met = guard.compute_auto_close_qty(sym, closed.c)
                if need_qty > 0 and guard.st.positions.get(sym, 0.0) > 0:
                    try:
                        # cap por balance base (venta) – en spot generalmente sí hay base para vender
                        capped_qty = min(need_qty, abs(guard.st.positions.get(sym, 0.0)))
                        if capped_qty > 0:
                            resp = await exec_adapter.place_order(sym, "sell", "market", capped_qty, mark_price=closed.c)
                            guard.update_position_on_order(sym, "sell", capped_qty)
                            # reconstruye la pos actual del guard como Position (spot)
                            pos = Position(
                                qty=guard.st.positions.get(sym, 0.0),
                                avg_price=guard.st.prices.get(sym, closed.c),  # aproximación si no hay avg
                                realized_pnl=0.0,  # lo traemos de DB si quieres; por simplicidad acumulamos aquí
                                fees_paid=0.0
                            )
                            # aplica fill a coste de mercado (closed.c) con fee taker (usa 7.5 bps por defecto spot testnet)
                            taker_fee_bps = 7.5
                            pos = apply_fill(pos, side, float(capped_qty), float(closed.c), taker_fee_bps,
                                             venue_type="spot")

                            # guarda posición y snapshot
                            upsert_position(engine, venue="binance_spot_testnet", symbol=sym, qty=pos.qty,
                                            avg_price=pos.avg_price, realized_pnl=pos.realized_pnl,
                                            fees_paid=pos.fees_paid)
                            upnl = mark_to_market(pos, closed.c)
                            insert_pnl_snapshot(engine, venue="binance_spot_testnet", symbol=sym, qty=pos.qty,
                                                price=closed.c, avg_price=pos.avg_price, upnl=upnl,
                                                rpnl=pos.realized_pnl, fees=pos.fees_paid)
                            log.warning("[%s] AUTO_CLOSE spot: %s | resp=%s", sym, msg, resp)
                            if engine is not None:
                                try:
                                    insert_risk_event(engine, venue="binance_spot_testnet", symbol=sym,
                                                      kind="AUTO_CLOSE", message=msg,
                                                      details={"requested_close_qty": need_qty, "executed_qty": capped_qty})
                                except Exception:
                                    pass
                    except Exception as e:
                        log.error("[%s] AUTO_CLOSE spot falló: %s", sym, e)
            continue  # no enviamos la orden “normal” si había violación

        # Balance → cap qty
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

        try:
            resp = await exec_adapter.place_order(sym, side, "market", capped_qty, mark_price=closed.c)
            log.info("[%s] SPOT TESTNET FILL %s", sym, resp)
            guard.update_position_on_order(sym, side, capped_qty)
            if engine is not None:
                try:
                    insert_order(engine,
                                 strategy="breakout_atr_spot",
                                 exchange="binance_spot_testnet",
                                 symbol=sym,
                                 side=side,
                                 type_="market",
                                 qty=float(capped_qty),
                                 px=None,
                                 status=resp.get("status", "sent"),
                                 ext_order_id=str(resp.get("resp", {}).get("id") if isinstance(resp.get("resp"), dict) else None),
                                 notes=resp)
                except Exception as e:
                    log.debug("[%s] No se pudo insertar order: %s", sym, e)
        except Exception as e:
            log.error("[%s] Error orden spot testnet: %s", sym, e)
