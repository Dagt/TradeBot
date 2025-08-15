# src/tradingbot/live/runner_spot_testnet.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
import pandas as pd

from ..execution.normalize import adjust_order

from .runner import BarAggregator  # el mismo agregador de 1m
from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager


# Persistencia opcional
try:
    from ..storage.timescale import get_engine, insert_order, insert_bar_1m
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

async def run_live_binance_spot_testnet(
    symbol: str = "BTC/USDT",
    trade_qty: float = 0.001,
    persist_pg: bool = False
):
    """
    WS spot testnet -> barras 1m -> BreakoutATR -> Risk -> BinanceSpotAdapter (orden real testnet)
    """
    ws = BinanceSpotWSAdapter()
    exec_adapter = BinanceSpotAdapter()
    agg = BarAggregator()
    strat = BreakoutATR()
    risk = RiskManager(max_pos=1.0)

    engine = get_engine() if (persist_pg and _CAN_PG) else None

    async for t in ws.stream_trades(symbol):
        ts: datetime = t["ts"] or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t["qty"] or 0.0)

        closed = agg.on_trade(ts, px, qty)
        if closed is None:
            continue

        # Persistir barra 1m (opcional)
        if engine is not None:
            try:
                insert_bar_1m(engine, exchange="binance_spot_testnet", symbol=symbol,
                              ts=closed.ts_open, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
            except Exception as e:
                log.debug("No se pudo insertar barra 1m: %s", e)

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

            resp = await exec_adapter.place_order(symbol, side, "market", adj_qty, mark_price=adj_price)
            log.info("SPOT TESTNET FILL %s", resp)

            # Actualiza RiskManager con la cantidad realmente enviada
            try:
                risk.add_fill(side, adj_qty)
            except Exception:
                pass

        except Exception:
            log.exception("SPOT fallo al ENVIAR orden al exchange (place_order)")
            continue  # seguimos con el siguiente tick/iteración del loop

        # --- Persistencia (BD) ---
        if engine is not None:
            try:
                insert_order(
                    engine,
                    strategy="breakout_atr_spot",
                    exchange="binance_spot_testnet",
                    symbol=symbol,
                    side=side,
                    type_="market",
                    qty=float(adj_qty),
                    px=None,
                    status=resp.get("status", "sent") if isinstance(resp, dict) else "sent",
                    ext_order_id=resp.get("ext_order_id"),
                    notes=resp,
                )
            except Exception:
                log.exception("SPOT persist failure: insert_order")
