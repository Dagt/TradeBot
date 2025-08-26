from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Type, List, Any

import pandas as pd

from .runner import BarAggregator
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager, load_positions
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.correlation_service import CorrelationService
from ..risk.service import RiskService
from ..execution.paper import PaperAdapter
from ..risk.oco import OcoBook, load_active_oco

from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..adapters.binance_ws import BinanceWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..adapters.bybit_spot import BybitSpotAdapter as BybitSpotWSAdapter, BybitSpotAdapter
from ..adapters.okx_spot import OKXSpotAdapter as OKXSpotWSAdapter, OKXSpotAdapter
from ..adapters.bybit_futures import BybitFuturesAdapter
from ..adapters.okx_futures import OKXFuturesAdapter

try:
    from ..storage.timescale import get_engine
    _CAN_PG = True
except Exception:  # pragma: no cover
    _CAN_PG = False

log = logging.getLogger(__name__)

AdapterTuple = Tuple[Callable[[], any], Callable[..., any], str]
ADAPTERS: Dict[Tuple[str, str], AdapterTuple] = {
    ("binance", "spot"): (BinanceSpotWSAdapter, BinanceSpotAdapter, "binance_spot_testnet"),
    ("binance", "futures"): (BinanceWSAdapter, BinanceFuturesAdapter, "binance_futures_testnet"),
    ("bybit", "spot"): (BybitSpotWSAdapter, BybitSpotAdapter, "bybit_spot_testnet"),
    ("bybit", "futures"): (BybitFuturesAdapter, BybitFuturesAdapter, "bybit_futures_testnet"),
    ("okx", "spot"): (OKXSpotWSAdapter, OKXSpotAdapter, "okx_spot_testnet"),
    ("okx", "futures"): (OKXFuturesAdapter, OKXFuturesAdapter, "okx_futures_testnet"),
}

@dataclass
class _SymbolConfig:
    symbol: str
    trade_qty: float

async def _run_symbol(exchange: str, market: str, cfg: _SymbolConfig, leverage: int,
                      dry_run: bool, total_cap_pct: float, per_symbol_cap_pct: float,
                      soft_cap_pct: float, soft_cap_grace_sec: int,
                      daily_max_loss_pct: float, daily_max_drawdown_pct: float,
                      corr_threshold: float,
                      config_path: str | None = None) -> None:
    ws_cls, exec_cls, venue = ADAPTERS[(exchange, market)]
    ws_kwargs: Dict[str, Any] = {"testnet": True}
    exec_kwargs: Dict[str, Any] = {"testnet": True}
    if market == "futures":
        exec_kwargs["leverage"] = leverage
    try:
        ws = ws_cls(**ws_kwargs)
    except TypeError:
        ws = ws_cls()
    try:
        exec_adapter = exec_cls(**exec_kwargs) if exec_kwargs else exec_cls()
    except TypeError:
        exec_kwargs.pop("leverage", None)
        try:
            exec_adapter = exec_cls(**exec_kwargs)
        except TypeError:
            exec_adapter = exec_cls()
    agg = BarAggregator()
    strat = BreakoutATR(config_path=config_path)
    risk_core = RiskManager(equity_pct=1.0)
    guard = PortfolioGuard(GuardConfig(
        total_cap_pct=total_cap_pct,
        per_symbol_cap_pct=per_symbol_cap_pct,
        venue=venue,
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec,
    ))
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_pct=daily_max_loss_pct,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        halt_action="close_all",
    ), venue=venue)
    corr = CorrelationService()
    risk = RiskService(risk_core, guard, dguard, corr_service=corr)
    broker = PaperAdapter(fee_bps=1.5)
    try:
        guard.refresh_usd_caps(broker.equity({}))
    except Exception:
        guard.refresh_usd_caps(0.0)
    engine = get_engine() if _CAN_PG else None
    oco_book = OcoBook()
    if engine is not None:
        pos_map = load_positions(engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(guard.cfg.venue, sym, data.get("qty", 0.0))
            risk.rm._entry_price = data.get("avg_price")
        oco_book.preload(
            load_active_oco(engine, venue=guard.cfg.venue, symbols=[cfg.symbol])
        )

    async for t in ws.stream_trades(cfg.symbol):
        ts: datetime = t.get("ts") or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t.get("qty") or 0.0)
        broker.update_last_price(cfg.symbol, px)
        risk.mark_price(cfg.symbol, px)
        risk.update_correlation(corr_threshold)
        halted, reason = risk.daily_mark(broker, cfg.symbol, px, 0.0)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break
        closed = agg.on_trade(ts, px, qty)
        if closed is None:
            continue
        df: pd.DataFrame = agg.last_n_bars_df(200)
        if len(df) < 140:
            continue
        sig = strat.on_bar({"window": df})
        if sig is None:
            continue
        allowed, reason, delta = risk.check_order(
            cfg.symbol,
            sig.side,
            closed.c,
            strength=sig.strength,
            corr_threshold=corr_threshold,
        )
        if not allowed or abs(delta) <= 0:
            if reason:
                log.warning("[PG] Bloqueado %s: %s", cfg.symbol, reason)
            continue
        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)
        if market == "futures" and dry_run:
            resp = await broker.place_order(cfg.symbol, side, "market", qty)
        else:
            resp = await exec_adapter.place_order(
                cfg.symbol, side, "market", qty, mark_price=closed.c
            )
        log.info("LIVE FILL %s", resp)
        risk.on_fill(cfg.symbol, side, qty, venue=venue if not dry_run else "paper")

async def run_live_testnet(
    exchange: str = "binance",
    market: str = "spot",
    symbols: List[str] | None = None,
    trade_qty: float = 0.001,
    leverage: int = 1,
    dry_run: bool = False,
    total_cap_pct: float = 1.0,
    per_symbol_cap_pct: float = 0.5,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    daily_max_loss_pct: float = 0.05,
    daily_max_drawdown_pct: float = 0.05,
    corr_threshold: float = 0.8,
    *,
    config_path: str | None = None,
) -> None:
    """Run a simple live loop on a crypto exchange testnet."""
    if (exchange, market) not in ADAPTERS:
        raise ValueError(f"Unsupported combination {exchange} {market}")
    symbols = symbols or ["BTC/USDT"]
    cfgs = [_SymbolConfig(symbol=s.upper().replace("-", "/"), trade_qty=trade_qty) for s in symbols]
    tasks = [
        _run_symbol(
            exchange,
            market,
            c,
            leverage,
            dry_run,
            total_cap_pct,
            per_symbol_cap_pct,
            soft_cap_pct,
            soft_cap_grace_sec,
            daily_max_loss_pct,
            daily_max_drawdown_pct,
            corr_threshold,
            config_path=config_path,
        )
        for c in cfgs
    ]
    await asyncio.gather(*tasks)
