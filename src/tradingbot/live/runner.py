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
from ..execution.router import ExecutionRouter
from ..strategies import STRATEGIES
from ..risk.manager import RiskManager, load_positions
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.service import RiskService
from ..broker.broker import Broker
from ..config import settings

# Persistencia opcional (Timescale). No es obligatorio para correr.
try:
    from ..storage.timescale import get_engine, insert_trade
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
    risk_pct: float = 0.0,
    total_cap_pct: float = 1.0,
    per_symbol_cap_pct: float = 0.5,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    daily_max_loss_pct: float = 0.05,
    daily_max_drawdown_pct: float = 0.05,
    *,
    strategy_name: str = "breakout_atr",
    config_path: str | None = None,
    params: dict | None = None,
) -> None:
    """
    Pipeline en vivo:
      WS Binance -> agregador 3m -> strategy -> Risk -> PaperAdapter
    """
    adapter = BinanceWSAdapter()
    broker = PaperAdapter(fee_bps=fee_bps)
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    params = params or {}
    strat = strat_cls(config_path=config_path, **params) if (config_path or params) else strat_cls()
    router = ExecutionRouter(
        [broker],
        prefer="maker",
        on_partial_fill=strat.on_partial_fill,
        on_order_expiry=strat.on_order_expiry,
    )
    exec_broker = Broker(router)
    guard = PortfolioGuard(GuardConfig(
        total_cap_pct=total_cap_pct,
        per_symbol_cap_pct=per_symbol_cap_pct,
        venue="binance",
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec,
    ))
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_pct=daily_max_loss_pct,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        halt_action="close_all",
    ), venue="binance")
    pg_engine = get_engine() if (persist_pg and _CAN_PG) else None
    risk = RiskService(
        guard,
        dguard,
        engine=pg_engine,
        account=broker.account,
        risk_pct=risk_pct,
    )
    risk.rm.allow_short = False
    guard.refresh_usd_caps(broker.equity({}))
    if pg_engine is not None:
        pos_map = load_positions(pg_engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(
                guard.cfg.venue, sym, data.get("qty", 0.0), entry_price=data.get("avg_price")
            )
    if persist_pg and not _CAN_PG:
        log.warning("Persistencia Timescale no disponible (sqlalchemy/psycopg2 no cargados).")

    agg = BarAggregator()
    fills = 0

    tick = getattr(settings, "tick_size", 0.0)

    def _limit_price(side: str) -> float:
        book = getattr(broker.state, "order_book", {}).get(symbol, {})
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        last = broker.state.last_px.get(symbol, 0.0)
        best_bid = bids[0][0] if bids else last
        best_ask = asks[0][0] if asks else last
        return (best_ask - tick) if side == "buy" else (best_bid + tick)

    async for t in adapter.stream_trades(symbol):
        ts: datetime = t["ts"] or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t["qty"] or 0.0)
        broker.update_last_price(symbol, px)
        risk.mark_price(symbol, px)
        halted, reason = risk.daily_mark(broker, symbol, px, 0.0)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break

        pos_qty, _ = risk.account.current_exposure(symbol)
        trade = risk.get_trade(symbol)
        if trade and abs(pos_qty) > risk.rm.min_order_qty:
            risk.update_trailing(trade, px)
            decision = risk.manage_position(trade)
            if decision == "close":
                close_side = "sell" if pos_qty > 0 else "buy"
                prev_rpnl = broker.state.realized_pnl
                price = _limit_price(close_side)
                resp = await exec_broker.place_limit(
                    symbol,
                    close_side,
                    price,
                    abs(pos_qty),
                    on_partial_fill=strat.on_partial_fill,
                    on_order_expiry=strat.on_order_expiry,
                )
                risk.on_fill(symbol, close_side, abs(pos_qty), venue="binance")
                delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                if halted:
                    log.error("[HALT] motivo=%s", reason)
                    break
                continue
            if decision in {"scale_in", "scale_out"}:
                target = risk.calc_position_size(trade.get("strength", 1.0), px)
                delta_qty = target - abs(pos_qty)
                if abs(delta_qty) > risk.rm.min_order_qty:
                    side = trade["side"] if delta_qty > 0 else ("sell" if trade["side"] == "buy" else "buy")
                    prev_rpnl = broker.state.realized_pnl
                    price = _limit_price(side)
                    resp = await exec_broker.place_limit(
                        symbol,
                        side,
                        price,
                        abs(delta_qty),
                        on_partial_fill=strat.on_partial_fill,
                        on_order_expiry=strat.on_order_expiry,
                    )
                    risk.on_fill(symbol, side, abs(delta_qty), venue="binance")
                    delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                    halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                    if halted:
                        log.error("[HALT] motivo=%s", reason)
                        break
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

        pending = getattr(strat, "pending_qty", {}).get(symbol, 0.0)
        allowed, reason, delta = risk.check_order(
            symbol,
            signal.side,
            closed.c,
            strength=signal.strength,
            pending_qty=pending,
        )
        if not allowed or abs(delta) <= 1e-9:
            if not allowed:
                log.warning("[RISK] Bloqueado %s: %s", symbol, reason)
            continue

        side = "buy" if delta > 0 else "sell"
        prev_rpnl = broker.state.realized_pnl
        price = signal.limit_price if signal.limit_price is not None else _limit_price(side)
        resp = await exec_broker.place_limit(
            symbol,
            side,
            price,
            abs(delta),
            on_partial_fill=strat.on_partial_fill,
            on_order_expiry=strat.on_order_expiry,
        )
        fills += 1
        log.info("FILL live %s", resp)
        risk.on_fill(symbol, side, abs(delta), venue="binance")
        delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
        halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break

        # Persistir barra 3m si quieres (opcional; requiere tabla market.bars creada)
        # Puedes descomentar y ampliar con INSERT a bars.
        # if pg_engine is not None:
        #     try:
        #         with pg_engine.begin() as conn:
        #             conn.execute(
        #                 text("""INSERT INTO market.bars (ts,timeframe,exchange,symbol,o,h,l,c,v)
        #                        VALUES (:ts,'3m','binance',:sym,:o,:h,:l,:c,:v)"""),
        #                 dict(ts=closed.ts_open, sym=symbol, o=closed.o, h=closed.h, l=closed.l, c=closed.c, v=closed.v)
        #             )
        #     except Exception as e:
        #         log.debug("No se pudo insertar barra: %s", e)

        # Logear equity/pnl cada barra cerrada
        eq = broker.equity()
        log.info("Bar close %s | px=%.2f vol=%.4f | fills=%d | equity=%.4f",
                 closed.ts_open.isoformat(), closed.c, closed.v, fills, eq)
