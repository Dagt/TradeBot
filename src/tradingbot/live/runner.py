# src/tradingbot/live/runner.py
from __future__ import annotations
import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

import pandas as pd

from ..adapters.binance_ws import BinanceWSAdapter
from ..execution.paper import PaperAdapter
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig

# Persistencia opcional (Timescale). No es obligatorio para correr.
try:
    from ..storage.timescale import get_engine, insert_trade, insert_risk_event
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

@dataclass
class Bar:
    ts_open: datetime
    o: float
    h: float
    l: float
    c: float
    v: float

class BarAggregator:
    """
    Agrega trades a barras de 1 minuto por timestamp (UTC).
    Cierra barra cuando cambia el minuto ("mm") del trade entrante.
    """
    def __init__(self):
        self.current_minute: Optional[datetime] = None
        self.cur: Optional[Bar] = None
        self.completed: Deque[Bar] = deque(maxlen=5000)

    def on_trade(self, ts: datetime, px: float, qty: float) -> Optional[Bar]:
        # Normaliza a minuto: YYYY-mm-dd HH:MM:00 UTC
        minute = ts.replace(second=0, microsecond=0)
        if self.current_minute is None:
            self.current_minute = minute
            self.cur = Bar(ts_open=minute, o=px, h=px, l=px, c=px, v=qty)
            return None

        if minute == self.current_minute:
            b = self.cur
            if px > b.h: b.h = px
            if px < b.l: b.l = px
            b.c = px
            b.v += qty
            return None
        else:
            # Se cerró la barra anterior
            closed = self.cur
            self.completed.append(closed)
            # nueva barra
            self.current_minute = minute
            self.cur = Bar(ts_open=minute, o=px, h=px, l=px, c=px, v=qty)
            return closed

    def last_n_bars_df(self, n: int) -> pd.DataFrame:
        bars = list(self.completed)[-n:]
        if not bars:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        return pd.DataFrame({
            "timestamp": [int(b.ts_open.timestamp()) for b in bars],
            "open": [b.o for b in bars],
            "high": [b.h for b in bars],
            "low": [b.l for b in bars],
            "close": [b.c for b in bars],
            "volume": [b.v for b in bars],
        })

async def run_live_binance(
    symbol: str = "BTC/USDT",
    fee_bps: float = 1.5,
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
    Pipeline en vivo:
      WS Binance -> agregador 1m -> BreakoutATR -> Risk -> PaperAdapter
    """
    adapter = BinanceWSAdapter()
    broker = PaperAdapter(fee_bps=fee_bps)
    risk = RiskManager(max_pos=1.0)
    strat = BreakoutATR()
    guard = PortfolioGuard(GuardConfig(
        total_cap_usdt=total_cap_usdt,
        per_symbol_cap_usdt=per_symbol_cap_usdt,
        venue="binance",
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec,
    ))
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        halt_action="close_all",
    ), venue="binance")

    pg_engine = get_engine() if (persist_pg and _CAN_PG) else None
    if persist_pg and not _CAN_PG:
        log.warning("Persistencia Timescale no disponible (sqlalchemy/psycopg2 no cargados).")

    agg = BarAggregator()
    fills = 0

    async for t in adapter.stream_trades(symbol):
        ts: datetime = t["ts"] or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t["qty"] or 0.0)
        broker.update_last_price(symbol, px)
        guard.mark_price(symbol, px)
        dguard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={symbol: px}))
        halted, reason = dguard.check_halt()
        if halted:
            log.error("[HALT] motivo=%s", reason)
            if pg_engine is not None:
                try:
                    insert_risk_event(pg_engine, venue="binance", symbol=symbol,
                                      kind=f"HALT_{reason}", message=f"HALT: {reason}", details={})
                except Exception:
                    log.exception("Persist failure: risk_event HALT")
            continue

        # Persistencia opcional: guardar trade crudo
        if pg_engine is not None:
            try:
                class TickObj:  # tiny proxy para insert_trade existente
                    exchange = "binance"
                    symbol = symbol
                    price = px
                    qty = qty
                    side = t.get("side")
                insert_trade(pg_engine, TickObj)
            except Exception as e:
                log.debug("No se pudo insertar trade: %s", e)

        closed = agg.on_trade(ts, px, qty)
        if closed is None:
            continue  # seguimos acumulando la barra en curso

        # Tenemos una barra cerrada -> formamos ventana para la estrategia
        df = agg.last_n_bars_df(200)  # ventana para ATR/EMA
        if len(df) < 140:  # warmup básico
            continue

        bar = {"window": df}
        signal = strat.on_bar(bar)
        if signal is None:
            continue

        delta = risk.size(signal.side, signal.strength)
        if abs(delta) > 1e-9:
            side = "buy" if delta > 0 else "sell"
            action, reason, metrics = guard.soft_cap_decision(symbol, side, abs(delta), closed.c)
            if action == "block":
                log.warning("[PG] Bloqueado %s: %s", symbol, reason)
                if pg_engine is not None:
                    try:
                        insert_risk_event(pg_engine, venue="binance", symbol=symbol,
                                          kind="VIOLATION", message=reason, details=metrics)
                    except Exception:
                        log.exception("Persist failure: risk_event VIOLATION")
                continue
            elif action == "soft_allow" and pg_engine is not None:
                try:
                    insert_risk_event(pg_engine, venue="binance", symbol=symbol,
                                      kind="INFO", message=reason, details=metrics)
                except Exception:
                    log.exception("Persist failure: risk_event INFO")
            resp = await broker.place_order(symbol, side, "market", abs(delta))
            fills += 1
            log.info("FILL live %s", resp)
            risk.add_fill(side, abs(delta))
            guard.update_position_on_order(symbol, side, abs(delta))

        # Persistir barra 1m si quieres (opcional; requiere tabla market.bars creada)
        # Puedes descomentar y ampliar con INSERT a bars.
        # if pg_engine is not None:
        #     try:
        #         with pg_engine.begin() as conn:
        #             conn.execute(
        #                 text("""INSERT INTO market.bars (ts,timeframe,exchange,symbol,o,h,l,c,v)
        #                        VALUES (:ts,'1m','binance',:sym,:o,:h,:l,:c,:v)"""),
        #                 dict(ts=closed.ts_open, sym=symbol, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
        #             )
        #     except Exception as e:
        #         log.debug("No se pudo insertar barra: %s", e)

        # Logear equity/pnl cada barra cerrada
        eq = broker.equity()
        log.info("Bar close %s | px=%.2f vol=%.4f | fills=%d | equity=%.4f",
                 closed.ts_open.isoformat(), closed.c, closed.v, fills, eq)
