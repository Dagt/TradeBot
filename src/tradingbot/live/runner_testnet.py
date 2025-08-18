from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Type, List, Any

import pandas as pd

from .runner import BarAggregator
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..execution.paper import PaperAdapter

from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..adapters.binance_ws import BinanceWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..adapters.bybit_spot import BybitSpotAdapter as BybitSpotWSAdapter, BybitSpotAdapter
from ..adapters.okx_spot import OKXSpotAdapter as OKXSpotWSAdapter, OKXSpotAdapter
from ..adapters.bybit_futures import BybitFuturesAdapter
from ..adapters.okx_futures import OKXFuturesAdapter

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
                      dry_run: bool, total_cap_usdt: float, per_symbol_cap_usdt: float,
                      soft_cap_pct: float, soft_cap_grace_sec: int,
                      daily_max_loss_usdt: float, daily_max_drawdown_pct: float,
                      max_consecutive_losses: int, config_path: str | None = None) -> None:
    ws_cls, exec_cls, venue = ADAPTERS[(exchange, market)]
    ws_kwargs: Dict[str, Any] = {}
    exec_kwargs: Dict[str, Any] = {}
    if market == "futures":
        ws_kwargs["testnet"] = True
        exec_kwargs.update({"leverage": leverage, "testnet": True})
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
    risk = RiskManager(max_pos=1.0)
    guard = PortfolioGuard(GuardConfig(
        total_cap_usdt=total_cap_usdt,
        per_symbol_cap_usdt=per_symbol_cap_usdt,
        venue=venue,
        soft_cap_pct=soft_cap_pct,
        soft_cap_grace_sec=soft_cap_grace_sec,
    ))
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        halt_action="close_all",
    ), venue=venue)
    broker = PaperAdapter(fee_bps=1.5)

    async for t in ws.stream_trades(cfg.symbol):
        ts: datetime = t.get("ts") or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t.get("qty") or 0.0)
        broker.update_last_price(cfg.symbol, px)
        guard.mark_price(cfg.symbol, px)
        dguard.on_mark(datetime.now(timezone.utc), equity_now=broker.equity(mark_prices={cfg.symbol: px}))
        halted, reason = dguard.check_halt(broker)
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
        delta = risk.size(sig.side, sig.strength)
        if abs(delta) < 1e-9:
            continue
        side = "buy" if delta > 0 else "sell"
        if not risk.check_limits(closed.c):
            log.warning("RiskManager disabled; kill switch activated")
            continue
        action, reason, _ = guard.soft_cap_decision(cfg.symbol, side, abs(cfg.trade_qty), closed.c)
        if action == "block":
            log.warning("[PG] Bloqueado %s: %s", cfg.symbol, reason)
            continue
        if market == "futures" and dry_run:
            resp = await broker.place_order(cfg.symbol, side, "market", cfg.trade_qty)
        else:
            resp = await exec_adapter.place_order(cfg.symbol, side, "market", cfg.trade_qty, mark_price=closed.c)
        log.info("LIVE FILL %s", resp)
        risk.add_fill(side, cfg.trade_qty)

async def run_live_testnet(
    exchange: str = "binance",
    market: str = "spot",
    symbols: List[str] | None = None,
    trade_qty: float = 0.001,
    leverage: int = 1,
    dry_run: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    daily_max_loss_usdt: float = 100.0,
    daily_max_drawdown_pct: float = 0.05,
    max_consecutive_losses: int = 3,
    *,
    config_path: str | None = None,
) -> None:
    """Run a simple live loop on a crypto exchange testnet."""
    if (exchange, market) not in ADAPTERS:
        raise ValueError(f"Unsupported combination {exchange} {market}")
    symbols = symbols or ["BTC/USDT"]
    cfgs = [_SymbolConfig(symbol=s.upper().replace("-", "/"), trade_qty=trade_qty) for s in symbols]
    tasks = [
        _run_symbol(exchange, market, c, leverage, dry_run, total_cap_usdt,
                    per_symbol_cap_usdt, soft_cap_pct, soft_cap_grace_sec,
                    daily_max_loss_usdt, daily_max_drawdown_pct,
                    max_consecutive_losses, config_path=config_path)
        for c in cfgs
    ]
    await asyncio.gather(*tasks)
