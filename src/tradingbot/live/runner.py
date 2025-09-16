# src/tradingbot/live/runner.py
from __future__ import annotations
import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, Optional
import time

import ccxt
import pandas as pd

from ..adapters.binance import BinanceWSAdapter
from ..execution.paper import PaperAdapter
from ..execution.router import ExecutionRouter
from ..strategies import STRATEGIES
from ..risk.service import load_positions
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.service import RiskService
from ..broker.broker import Broker
from ..config import settings
from ..core.symbols import normalize
from ..utils.price import limit_price_from_close

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
    """Agrega trades en barras OHLCV para un timeframe configurable.

    El timeframe puede expresarse en minutos u horas (ej. ``"1m"``, ``"5m"``,
    ``"1h"`, ``"4h"``). Se cierra una barra cuando cambia el periodo calculado
    para el trade entrante.
    """

    def __init__(self, timeframe: str = "1m") -> None:
        self.timeframe = timeframe
        seconds = ccxt.Exchange.parse_timeframe(timeframe)
        if seconds % 60:
            raise ValueError("timeframe must resolve to a whole number of minutes")
        self._mins = int(seconds // 60) or 1
        self.current_period: Optional[datetime] = None
        self.cur: Optional[Bar] = None
        self.completed: Deque[Bar] = deque(maxlen=5000)

    def _floor_ts(self, ts: datetime) -> datetime:
        """Floor ``ts`` to the start of the current timeframe."""
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = timedelta(minutes=self._mins)
        return ts - ((ts - epoch) % delta)

    def on_trade(self, ts: datetime, px: float, qty: float) -> Optional[Bar]:
        period = self._floor_ts(ts)
        if self.current_period is None:
            self.current_period = period
            self.cur = Bar(ts_open=period, o=px, h=px, l=px, c=px, v=qty)
            return None

        if period == self.current_period:
            b = self.cur
            if px > b.h:
                b.h = px
            if px < b.l:
                b.l = px
            b.c = px
            b.v += qty
            return None
        else:
            # Se cerró la barra anterior
            closed = self.cur
            self.completed.append(closed)
            # nueva barra
            self.current_period = period
            self.cur = Bar(ts_open=period, o=px, h=px, l=px, c=px, v=qty)
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
    reprice_bps: float = 0.0,
    *,
    strategy_name: str = "breakout_atr",
    config_path: str | None = None,
    params: dict | None = None,
    timeframe: str = "1m",
    venue: str = "binance_futures",
) -> None:
    """
    Pipeline en vivo:
      WS Binance -> agregador 3m -> strategy -> Risk -> PaperAdapter
    """
    exchange, market = venue.split("_", 1)
    log.info("Connecting to Binance WS for %s", symbol)
    adapter = BinanceWSAdapter()
    broker = PaperAdapter(fee_bps=fee_bps)
    broker.account.market_type = market
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    params = params or {}
    strat = strat_cls(config_path=config_path, **params) if (config_path or params) else strat_cls()
    router = ExecutionRouter([
        broker
    ], prefer="maker")
    exec_broker = Broker(router)
    guard = PortfolioGuard(GuardConfig(
        total_cap_pct=total_cap_pct,
        per_symbol_cap_pct=per_symbol_cap_pct,
        venue=exchange,
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec,
    ))
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_pct=daily_max_loss_pct,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        halt_action="close_all",
    ), venue=exchange)
    pg_engine = get_engine() if (persist_pg and _CAN_PG) else None
    risk = RiskService(
        guard,
        dguard,
        engine=pg_engine,
        account=broker.account,
        risk_pct=risk_pct,
        market_type=market,
    )
    guard.refresh_usd_caps(broker.equity({}))
    strat.risk_service = risk
    if pg_engine is not None:
        pos_map = load_positions(pg_engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(
                guard.cfg.venue, sym, data.get("qty", 0.0), entry_price=data.get("avg_price")
            )
    if persist_pg and not _CAN_PG:
        log.warning("Persistencia Timescale no disponible (sqlalchemy/psycopg2 no cargados).")

    agg = BarAggregator(timeframe=timeframe)
    fills = 0

    tick = 0.0
    meta = None
    rest = getattr(adapter, "rest", None)
    if rest is not None and hasattr(rest, "meta"):
        meta = rest.meta
    elif hasattr(adapter, "meta"):
        meta = adapter.meta
    if meta is not None:
        try:
            fetch_symbol = None
            symbols = getattr(meta.client, "symbols", [])
            if symbols:
                fetch_symbol = next((s for s in symbols if normalize(s) == symbol), None)
            if fetch_symbol is None:
                fetch_symbol = symbol.replace("-", "/")
            rules = meta.rules_for(fetch_symbol)
            tick = float(getattr(rules, "price_step", 0.0) or 0.0)
        except Exception:
            tick = 0.0

    last_price = 0.0

    def _wrap_cb(orig_cb):
        def _cb(order, res):
            action = orig_cb(order, res) if orig_cb else "re_quote"
            if action not in {"re_quote", "requote", "re-quote"}:
                return action
            lp = order.price or res.get("price")
            if lp is None or last_price <= 0 or reprice_bps <= 0:
                order.price = limit_price_from_close(order.side, last_price, tick)
                return "re_quote"
            diff = abs(last_price - lp) / lp
            if diff > reprice_bps / 10000.0:
                order.price = limit_price_from_close(order.side, last_price, tick)
                return "re_quote"
            return None
        return _cb

    pf_cb = _wrap_cb(strat.on_partial_fill)
    oe_cb = _wrap_cb(strat.on_order_expiry)
    router.on_partial_fill = pf_cb
    router.on_order_expiry = oe_cb

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
        last_price = px
        events = broker.update_last_price(symbol, px)
        for ev in events or []:
            await router.handle_paper_event(ev)
        risk.mark_price(symbol, px)
        halted, reason = risk.daily_mark(broker, symbol, px, 0.0)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break

        pos_qty, _ = risk.account.current_exposure(symbol)
        trade = risk.get_trade(symbol)
        if trade and abs(pos_qty) > risk.min_order_qty:
            risk.update_trailing(trade, px)
            trade["_trail_done"] = True
            decision = risk.manage_position(trade)
            if decision == "close":
                close_side = "sell" if pos_qty > 0 else "buy"
                price = _limit_price(close_side)
                qty_close = abs(pos_qty)
                notional = qty_close * price
                if not risk.register_order(symbol, notional):
                    continue
                prev_rpnl = broker.state.realized_pnl
                resp = await exec_broker.place_limit(
                    symbol,
                    close_side,
                    price,
                    qty_close,
                    on_partial_fill=pf_cb,
                    on_order_expiry=oe_cb,
                )
                filled_qty = float(resp.get("filled_qty", 0.0))
                pending_qty = float(resp.get("pending_qty", 0.0))
                risk.account.update_open_order(symbol, close_side, pending_qty)
                risk.on_fill(symbol, close_side, filled_qty, venue="binance")
                if pending_qty > 0:
                    risk.account.update_open_order(symbol, close_side, -pending_qty)
                delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                if halted:
                    log.error("[HALT] motivo=%s", reason)
                    break
                continue
            if decision in {"scale_in", "scale_out"}:
                target = risk.calc_position_size(trade.get("strength", 1.0), px, clamp=False)
                delta_qty = target - abs(pos_qty)
                if abs(delta_qty) > risk.min_order_qty:
                    side = trade["side"] if delta_qty > 0 else ("sell" if trade["side"] == "buy" else "buy")
                    prev_rpnl = broker.state.realized_pnl
                    price = _limit_price(side)
                    qty_scale = abs(delta_qty)
                    notional = qty_scale * price
                    if not risk.register_order(symbol, notional):
                        continue
                    resp = await exec_broker.place_limit(
                        symbol,
                        side,
                        price,
                        qty_scale,
                        on_partial_fill=pf_cb,
                        on_order_expiry=oe_cb,
                    )
                    filled_qty = float(resp.get("filled_qty", 0.0))
                    pending_qty = float(resp.get("pending_qty", 0.0))
                    risk.account.update_open_order(symbol, side, pending_qty)
                    risk.on_fill(symbol, side, filled_qty, venue="binance")
                    if pending_qty > 0:
                        risk.account.update_open_order(symbol, side, -pending_qty)
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

        bar = {"window": df, "symbol": symbol}
        signal = strat.on_bar(bar)
        if signal is None:
            continue
        signal_ts = getattr(signal, "signal_ts", time.time())

        pending = getattr(strat, "pending_qty", {}).get(symbol, 0.0)
        allowed, reason, delta = risk.check_order(
            symbol,
            signal.side,
            closed.c,
            strength=signal.strength,
            pending_qty=pending,
            volatility=bar.get("atr") or bar.get("volatility"),
            target_volatility=bar.get("target_volatility"),
        )
        if not allowed or abs(delta) <= 1e-9:
            if not allowed and reason != "below_min_qty":
                log.warning("[RISK] Bloqueado %s: %s", symbol, reason)
            continue

        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)
        price = signal.limit_price if signal.limit_price is not None else _limit_price(side)
        notional = qty * price
        if not risk.register_order(symbol, notional):
            continue
        prev_rpnl = broker.state.realized_pnl
        resp = await exec_broker.place_limit(
            symbol,
            side,
            price,
            qty,
            on_partial_fill=pf_cb,
            on_order_expiry=oe_cb,
            signal_ts=signal_ts,
        )
        fills += 1
        log.info("FILL live %s", resp)
        filled_qty = float(resp.get("filled_qty", 0.0))
        pending_qty = float(resp.get("pending_qty", 0.0))
        risk.account.update_open_order(symbol, side, pending_qty)
        risk.on_fill(symbol, side, filled_qty, venue="binance")
        if pending_qty > 0:
            risk.account.update_open_order(symbol, side, -pending_qty)
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
