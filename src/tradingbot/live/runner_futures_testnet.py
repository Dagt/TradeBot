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
from ..strategies.breakout_vol import BreakoutVol
from ..risk.manager import RiskManager
from ..execution.order_types import Order
from ..execution.paper import PaperAdapter  # para equity mark si no quieres consultar mark price
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig

# Persistencia opcional
try:
    from ..storage.timescale import get_engine, insert_risk_event
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

@dataclass
class LiveState:
    fills: int = 0

async def run_live_binance_futures_testnet(
    symbol: str = "BTC/USDT",
    leverage: int = 5,
    trade_qty: float = 0.001,  # tamaño de orden por señal
    dry_run: bool = True,      # True = no envía a testnet; usa PaperAdapter
    persist_pg: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    daily_max_loss_usdt: float = 100.0,
    daily_max_drawdown_pct: float = 0.05,
    max_consecutive_losses: int = 3,
):
    """
    WS Binance (precio) -> agregación 1m -> estrategia -> Risk -> (Paper o Futures Testnet)
    """
    ws = BinanceWSAdapter()
    agg = BarAggregator()
    strat = BreakoutVol()
    risk = RiskManager(max_pos=1.0)
    state = LiveState()
    guard = PortfolioGuard(GuardConfig(
        total_cap_usdt=total_cap_usdt,
        per_symbol_cap_usdt=per_symbol_cap_usdt,
        venue="binance_futures_testnet",
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec
    ))
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        halt_action="close_all"
    ), venue="binance_futures_testnet")

    engine = get_engine() if (persist_pg and _CAN_PG) else None

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
        guard.mark_price(symbol, px)
        dguard.on_mark(datetime.now(timezone.utc), equity_now=0.0)
        halted, reason = dguard.check_halt()
        if halted:
            log.error("[HALT FUTURES] motivo=%s", reason)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue="binance_futures_testnet", symbol=symbol,
                                      kind=f"HALT_{reason}", message=f"FUTURES halt: {reason}", details={})
                except Exception:
                    log.exception("FUTURES persist failure: risk_event HALT")
            continue

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

        if not risk.check_limits(closed.c):
            log.warning("RiskManager disabled; kill switch activated")
            continue

        action, reason, metrics = guard.soft_cap_decision(symbol, side, abs(trade_qty), closed.c)
        if action == "block":
            log.warning("[PG] Bloqueado %s: %s", symbol, reason)
            if engine is not None:
                try:
                    insert_risk_event(engine, venue="binance_futures_testnet", symbol=symbol,
                                      kind="VIOLATION", message=reason, details=metrics)
                except Exception:
                    log.exception("FUTURES persist failure: risk_event VIOLATION")
            continue
        elif action == "soft_allow" and engine is not None:
            try:
                insert_risk_event(engine, venue="binance_futures_testnet", symbol=symbol,
                                  kind="INFO", message=reason, details=metrics)
            except Exception:
                log.exception("FUTURES persist failure: risk_event INFO")

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
            guard.update_position_on_order(symbol, side, float(capped_qty))
        except Exception as e:
            log.error("Error enviando orden: %s", e)

