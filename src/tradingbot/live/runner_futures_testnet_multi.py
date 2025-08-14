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
from ..risk.pnl import Position, apply_fill, mark_to_market
from ..storage.timescale import upsert_position, insert_pnl_snapshot

try:
    from ..storage.timescale import get_engine, insert_order, insert_bar_1m, insert_portfolio_snapshot, insert_risk_event
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
    auto_close: bool = True  # NEW
):
    symbols = [s.upper().replace("-", "/") for s in symbols]
    ws = BinanceFuturesWSAdapter()
    exec_adapter = BinanceFuturesAdapter(leverage=leverage, testnet=True)

    aggs = {s: BarAggregator() for s in symbols}
    strats = {s: BreakoutATR() for s in symbols}
    risks = {s: RiskManager(max_pos=1.0) for s in symbols}

    guard = PortfolioGuard(GuardConfig(total_cap_usdt=total_cap_usdt,
                                       per_symbol_cap_usdt=per_symbol_cap_usdt,
                                       venue="binance_futures_um_testnet"))

    engine = get_engine() if (persist_pg and _CAN_PG) else None

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
                insert_bar_1m(engine, exchange="binance_futures_um_testnet", symbol=sym,
                              ts=closed.ts_open, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
            except Exception as e:
                log.debug("No se pudo insertar barra 1m (%s): %s", sym, e)
            try:
                pos = guard.st.positions.get(sym, 0.0)
                notional = abs(pos) * closed.c
                insert_portfolio_snapshot(engine, venue="binance_futures_um_testnet",
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

        exceed, reason, metrics = guard.would_exceed_caps(sym, side, abs(trade_qty), closed.c)
        if exceed:
            log.info("[%s] Orden bloqueada por guardia: %s | metrics=%s", sym, reason, metrics)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue="binance_futures_um_testnet", symbol=sym,
                                      kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    pass

            # AUTO‑CLOSE (Futures): reduce‑only opuesto a la señal y/o contra el signo de la posición
            if auto_close:
                need_qty, msg, met = guard.compute_auto_close_qty(sym, closed.c)
                if need_qty > 0:
                    try:
                        # Si la posición actual es >0 (larga), cerramos vendiendo reduce-only; si <0, compramos reduce-only
                        cur_pos = guard.st.positions.get(sym, 0.0)
                        close_side = "sell" if cur_pos > 0 else "buy"
                        resp = await exec_adapter.place_order(sym, close_side, "market", need_qty, reduce_only=True, mark_price=closed.c)
                        guard.update_position_on_order(sym, close_side, need_qty)
                        pos = Position(
                            qty=guard.st.positions.get(sym, 0.0),
                            avg_price=guard.st.prices.get(sym, closed.c),
                            realized_pnl=0.0,
                            fees_paid=0.0
                        )
                        taker_fee_bps = 2.0  # típico en futures testnet (ajústalo si usas otro)
                        pos = apply_fill(pos, side, float(capped_qty), float(closed.c), taker_fee_bps,
                                         venue_type="futures")

                        upsert_position(engine, venue="binance_futures_um_testnet", symbol=sym, qty=pos.qty,
                                        avg_price=pos.avg_price, realized_pnl=pos.realized_pnl, fees_paid=pos.fees_paid)
                        upnl = mark_to_market(pos, closed.c)
                        insert_pnl_snapshot(engine, venue="binance_futures_um_testnet", symbol=sym, qty=pos.qty,
                                            price=closed.c, avg_price=pos.avg_price, upnl=upnl, rpnl=pos.realized_pnl,
                                            fees=pos.fees_paid)
                        log.warning("[%s] AUTO_CLOSE futures: %s | resp=%s", sym, msg, resp)
                        if engine is not None:
                            try:
                                insert_risk_event(engine, venue="binance_futures_um_testnet", symbol=sym,
                                                  kind="AUTO_CLOSE", message=msg,
                                                  details={"executed_side": close_side, "executed_qty": need_qty})
                            except Exception:
                                pass
                    except Exception as e:
                        log.error("[%s] AUTO_CLOSE futures falló: %s", sym, e)
            continue

        # Cap por balance USDT*lev
        try:
            bal = await fetch_futures_usdt_free(exec_adapter.rest)
            capped_qty = cap_qty_by_balance_futures(side, abs(trade_qty), closed.c, bal.free_usdt, leverage, utilization=0.5)
            if capped_qty <= 0:
                log.info("[%s] Skip order (USDT libre insuficiente). Free: %.4f", sym, bal.free_usdt)
                continue
        except Exception as e:
            log.warning("[%s] No se pudo leer balance futures: %s (continuo qty original)", sym, e)
            capped_qty = abs(trade_qty)

        try:
            resp = await exec_adapter.place_order(sym, side, "market", capped_qty, mark_price=closed.c)
            log.info("[%s] FUTURES TESTNET FILL %s", sym, resp)
            guard.update_position_on_order(sym, side, capped_qty)
            if engine is not None:
                try:
                    insert_order(engine,
                                 strategy="breakout_atr_futures",
                                 exchange="binance_futures_um_testnet",
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
            log.error("[%s] Error orden futures testnet: %s", sym, e)
