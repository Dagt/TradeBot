from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone

import uvicorn

from .runner import BarAggregator
from ..adapters.binance_ws import BinanceWSAdapter
from ..execution.order_types import Order
from ..execution.paper import PaperAdapter
from ..execution.router import ExecutionRouter
from ..risk.manager import RiskManager, load_positions
from ..risk.portfolio_guard import GuardConfig, PortfolioGuard
from ..risk.service import RiskService
from ..risk.correlation_service import CorrelationService
from ..risk.oco import OcoBook, load_active_oco
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
) -> None:
    """Run a simple live pipeline entirely in paper mode."""

    adapter = BinanceWSAdapter()
    broker = PaperAdapter()
    router = ExecutionRouter([broker])

    risk_core = RiskManager(equity_pct=1.0, equity_actual=1.0)
    guard = PortfolioGuard(GuardConfig(total_cap_usdt=1000.0, per_symbol_cap_usdt=500.0, venue="paper"))
    corr = CorrelationService()
    risk = RiskService(risk_core, guard, corr_service=corr)
    engine = get_engine() if _CAN_PG else None
    oco_book = OcoBook()
    if engine is not None:
        pos_map = load_positions(engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(guard.cfg.venue, sym, data.get("qty", 0.0))
            risk.rm._entry_price = data.get("avg_price")
        oco_book.preload(load_active_oco(engine, venue=guard.cfg.venue, symbols=[symbol]))

    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    strat = strat_cls(config_path=config_path) if config_path else strat_cls()

    server = await _start_metrics(metrics_port)

    agg = BarAggregator()
    try:
        async for t in adapter.stream_trades(symbol):
            ts = t.get("ts") or datetime.now(timezone.utc)
            px = float(t.get("price"))
            qty = float(t.get("qty", 0.0))
            broker.update_last_price(symbol, px)
            risk.mark_price(symbol, px)
            risk.update_correlation(corr_threshold)
            closed = agg.on_trade(ts, px, qty)
            if closed is None:
                continue
            df = agg.last_n_bars_df(200)
            if len(df) < 140:
                continue
            signal = strat.on_bar({"window": df})
            if signal is None:
                continue
            allowed, _reason, delta = risk.check_order(
                symbol,
                signal.side,
                closed.c,
                strength=signal.strength,
                corr_threshold=corr_threshold,
            )
            if not allowed or abs(delta) <= 0:
                continue
            side = "buy" if delta > 0 else "sell"
            order = Order(
                symbol=symbol,
                side=side,
                type_="market",
                qty=abs(delta),
                reduce_only=signal.reduce_only,
            )
            await router.execute(order)
            risk.on_fill(symbol, side, abs(delta), venue="paper")
    finally:
        server.should_exit = True

