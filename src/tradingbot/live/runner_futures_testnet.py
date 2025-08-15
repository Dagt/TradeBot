# src/tradingbot/live/runner_futures_testnet.py
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

import pandas as pd

from .runner import BarAggregator  # reutilizamos el agregador 1m
from ..adapters.binance_ws import BinanceWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.order_types import Order
from ..execution.paper import PaperAdapter  # para equity mark si no quieres consultar mark price

log = logging.getLogger(__name__)

@dataclass
class LiveState:
    fills: int = 0

async def run_live_binance_futures_testnet(
    symbol: str = "BTC/USDT",
    leverage: int = 5,
    trade_qty: float = 0.001,  # tamaño de orden por señal
    dry_run: bool = True,      # True = no envía a testnet; usa PaperAdapter
):
    """
    WS Binance (precio) -> agregación 1m -> estrategia -> Risk -> (Paper o Futures Testnet)
    """
    ws = BinanceWSAdapter()
    agg = BarAggregator()
    strat = BreakoutATR()
    risk = RiskManager(max_pos=1.0)
    state = LiveState()

    # Broker
    if dry_run:
        broker = PaperAdapter(fee_bps=1.5)
        exec_adapter = broker
        log.warning("DRY-RUN ACTIVADO: NO se enviarán órdenes a Testnet; usando PaperAdapter.")
    else:
        futures = BinanceFuturesAdapter(leverage=leverage, testnet=True)
        exec_adapter = futures
        # PaperAdapter para equity mark local (opcional)
        broker = PaperAdapter(fee_bps=1.5)

    async for t in ws.stream_trades(symbol):
        ts: datetime = t["ts"] or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t["qty"] or 0.0)

        # actualiza last local (para equity en PaperAdapter)
        broker.update_last_price(symbol, px)

        closed = agg.on_trade(ts, px, qty)
        if closed is None:
            continue

        # Ventana de barras para features
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

        # --- NEW: cap por balance futures (USDT libre y leverage) ---
        from ..execution.balance import fetch_futures_usdt_free, cap_qty_by_balance_futures
        try:
            bal = await fetch_futures_usdt_free(exec_adapter.rest)
            capped_qty = cap_qty_by_balance_futures(side, abs(trade_qty), closed.c, bal.free_usdt, leverage,
                                                    utilization=0.5)
            if capped_qty <= 0:
                log.info("Skip order (USDT libre insuficiente). Free: %s", bal.free_usdt)
                continue
        except Exception as e:
            log.warning("No se pudo leer balance futures: %s (continuo con qty original)", e)
            capped_qty = abs(trade_qty)

        try:
            resp = await exec_adapter.place_order(symbol, side, "market", capped_qty, mark_price=closed.c)
            state.fills += 1
            log.info("LIVE FILL (dry_run=%s) %s", dry_run, resp)
            # Actualizar RiskManager con el fill ejecutado
            risk.add_fill(side, capped_qty)
        except Exception as e:
            log.error("Error enviando orden: %s", e)

