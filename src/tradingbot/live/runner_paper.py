from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
import time

import uvicorn

from .runner import BarAggregator
from ..adapters.binance_ws import BinanceWSAdapter
from ..execution.paper import PaperAdapter
from ..execution.router import ExecutionRouter
from ..broker.broker import Broker
from ..config import settings
from ..risk.service import load_positions
from ..risk.portfolio_guard import GuardConfig, PortfolioGuard
from ..risk.service import RiskService
from ..risk.correlation_service import CorrelationService
from ..strategies import STRATEGIES
from monitoring import panel

try:
    from ..storage.timescale import get_engine
    _CAN_PG = True
except Exception:  # pragma: no cover
    _CAN_PG = False

log = logging.getLogger(__name__)


async def _start_metrics(port: int) -> uvicorn.Server:
    """Launch the monitoring panel in the background."""
    config = uvicorn.Config(panel.app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    return server


async def run_paper(
    symbol: str = "BTC/USDT",
    strategy_name: str = "breakout_atr",
    *,
    config_path: str | None = None,
    metrics_port: int = 8000,
    corr_threshold: float = 0.8,
    risk_pct: float = 0.0,
    params: dict | None = None,
) -> None:
    """Run a simple live pipeline entirely in paper mode."""

    adapter = BinanceWSAdapter()
    broker = PaperAdapter()

    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=0.5, venue="paper"))
    guard.refresh_usd_caps(1000.0)
    corr = CorrelationService()
    risk = RiskService(
        guard,
        corr_service=corr,
        account=broker.account,
        risk_pct=risk_pct,
    )
    risk.allow_short = False
    engine = get_engine() if _CAN_PG else None
    if engine is not None:
        pos_map = load_positions(engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(
                guard.cfg.venue, sym, data.get("qty", 0.0), entry_price=data.get("avg_price")
            )

    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    params = params or {}
    strat = strat_cls(config_path=config_path, **params) if (config_path or params) else strat_cls()
    strat.risk_service = risk

    router = ExecutionRouter(
        [broker],
        prefer="maker",
        on_partial_fill=strat.on_partial_fill,
        on_order_expiry=strat.on_order_expiry,
    )
    exec_broker = Broker(router)

    server = await _start_metrics(metrics_port)

    agg = BarAggregator()
    tick = getattr(settings, "tick_size", 0.0)
    purge_interval = settings.risk_purge_minutes * 60.0
    last_purge = time.time()

    def _limit_price(side: str) -> float:
        book = getattr(broker.state, "order_book", {}).get(symbol, {})
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        last = broker.state.last_px.get(symbol, 0.0)
        best_bid = bids[0][0] if bids else last
        best_ask = asks[0][0] if asks else last
        return (best_ask - tick) if side == "buy" else (best_bid + tick)
    try:
        async for t in adapter.stream_trades(symbol):
            ts = t.get("ts") or datetime.now(timezone.utc)
            px = float(t.get("price"))
            qty = float(t.get("qty", 0.0))
            broker.update_last_price(symbol, px)
            risk.mark_price(symbol, px)
            if time.time() - last_purge >= purge_interval:
                risk.purge([symbol])
                last_purge = time.time()
            risk.update_correlation(corr._returns.corr(), corr_threshold)
            pos_qty, _ = risk.account.current_exposure(symbol)
            trade = risk.get_trade(symbol)
            if trade and abs(pos_qty) > risk.min_order_qty:
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
                    filled_qty = float(resp.get("filled_qty", 0.0))
                    pending_qty = float(resp.get("pending_qty", 0.0))
                    risk.account.update_open_order(symbol, filled_qty + pending_qty)
                    risk.on_fill(symbol, close_side, filled_qty, venue="paper")
                    delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                    halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                    if halted:
                        log.error("[HALT] motivo=%s", reason)
                        break
                    continue
                if decision in {"scale_in", "scale_out"}:
                    target = risk.calc_position_size(trade.get("strength", 1.0), px)
                    delta_qty = target - abs(pos_qty)
                    if abs(delta_qty) > risk.min_order_qty:
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
                        filled_qty = float(resp.get("filled_qty", 0.0))
                        pending_qty = float(resp.get("pending_qty", 0.0))
                        risk.account.update_open_order(symbol, filled_qty + pending_qty)
                        risk.on_fill(symbol, side, filled_qty, venue="paper")
                        delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                        halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                        if halted:
                            log.error("[HALT] motivo=%s", reason)
                            break
                    continue
            closed = agg.on_trade(ts, px, qty)
            if closed is None:
                continue
            df = agg.last_n_bars_df(200)
            if len(df) < 140:
                continue
            bar = {"window": df, "symbol": symbol}
            signal = strat.on_bar(bar)
            if signal is None:
                continue
            signal_ts = getattr(signal, "signal_ts", time.time())
            allowed, _reason, delta = risk.check_order(
                symbol,
                signal.side,
                closed.c,
                strength=signal.strength,
                volatility=bar.get("atr") or bar.get("volatility"),
                target_volatility=bar.get("target_volatility"),
            )
            if not allowed or abs(delta) <= 0:
                continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            price = getattr(signal, "limit_price", None)
            price = price if price is not None else _limit_price(side)
            notional = qty * price
            if not risk.register_order(symbol, notional):
                continue
            prev_rpnl = broker.state.realized_pnl
            resp = await exec_broker.place_limit(
                symbol,
                side,
                price,
                qty,
                on_partial_fill=strat.on_partial_fill,
                on_order_expiry=strat.on_order_expiry,
                signal_ts=signal_ts,
            )
            filled_qty = float(resp.get("filled_qty", 0.0))
            pending_qty = float(resp.get("pending_qty", 0.0))
            risk.account.update_open_order(symbol, filled_qty + pending_qty)
            risk.on_fill(symbol, side, filled_qty, venue="paper")
            delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
            halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
            if halted:
                log.error("[HALT] motivo=%s", reason)
                break
    finally:
        server.should_exit = True

