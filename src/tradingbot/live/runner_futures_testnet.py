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

        # Mapeo simple: si delta>0 -> buy trade_qty; si delta<0 -> sell trade_qty
        side = "buy" if delta > 0 else "sell"
        order = Order(symbol=symbol, side=side, type_="market", qty=abs(trade_qty))
        try:
            resp = await exec_adapter.place_order(order.symbol, order.side, order.type_, order.qty)
            state.fills += 1
            log.info("LIVE FILL (dry_run=%s) %s", dry_run, resp)
        except Exception as e:
            log.error("Error enviando orden: %s", e)

        eq = broker.equity()
        log.info("Bar close %s | px=%.2f vol=%.4f | fills=%d | equity=%.4f",
                 closed.ts_open.isoformat(), closed.c, closed.v, state.fills, eq)
