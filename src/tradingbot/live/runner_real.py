"""Minimal live runner for real exchange trading.

This module is intentionally lightweight and mirrors the structure of
``runner_testnet`` but connects to real exchange endpoints.  It requires the
user to acknowledge the danger of live trading via the
``--i-know-what-im-doing`` flag and pulls API credentials from the ``.env``
file through :mod:`tradingbot.config`.

The runner features a ``dry_run`` mode that executes orders using the
``PaperAdapter`` while still streaming real market data.  When ``dry_run`` is
``False`` orders are forwarded to the corresponding exchange adapter.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from .runner import BarAggregator
from ..config import settings
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


AdapterTuple = Tuple[Callable[[], Any], Callable[..., Any], str]


# Mapping of (exchange, market) to websocket adapter, execution adapter and
# venue name.  These are the real counterparts (no ``_testnet`` suffixes).
ADAPTERS: Dict[Tuple[str, str], AdapterTuple] = {
    ("binance", "spot"): (BinanceSpotWSAdapter, BinanceSpotAdapter, "binance_spot"),
    ("binance", "futures"): (BinanceWSAdapter, BinanceFuturesAdapter, "binance_futures"),
    ("bybit", "spot"): (BybitSpotWSAdapter, BybitSpotAdapter, "bybit_spot"),
    ("bybit", "futures"): (BybitFuturesAdapter, BybitFuturesAdapter, "bybit_futures"),
    ("okx", "spot"): (OKXSpotWSAdapter, OKXSpotAdapter, "okx_spot"),
    ("okx", "futures"): (OKXFuturesAdapter, OKXFuturesAdapter, "okx_futures"),
}


def _get_keys(exchange: str) -> Tuple[str | None, str | None]:
    """Return API key/secret pair for ``exchange`` from settings."""

    if exchange == "binance":
        return settings.binance_api_key, settings.binance_api_secret
    if exchange == "bybit":
        return settings.bybit_api_key, settings.bybit_api_secret
    if exchange == "okx":
        return settings.okx_api_key, settings.okx_api_secret
    return None, None


@dataclass
class _SymbolConfig:
    symbol: str
    trade_qty: float


async def _run_symbol(
    exchange: str,
    market: str,
    cfg: _SymbolConfig,
    leverage: int,
    dry_run: bool,
    total_cap_usdt: float,
    per_symbol_cap_usdt: float,
    soft_cap_pct: float,
    soft_cap_grace_sec: int,
    daily_max_loss_pct: float,
    daily_max_drawdown_pct: float,
    corr_threshold: float,
    config_path: str | None = None,
) -> None:
    ws_cls, exec_cls, venue = ADAPTERS[(exchange, market)]
    api_key, api_secret = _get_keys(exchange)
    exec_kwargs: Dict[str, Any] = {"api_key": api_key, "api_secret": api_secret}
    if market == "futures":
        exec_kwargs["leverage"] = leverage
    ws = ws_cls()
    exec_adapter = exec_cls(**exec_kwargs)
    agg = BarAggregator()
    strat = BreakoutATR(config_path=config_path)
    risk_core = RiskManager(equity_pct=1.0)
    guard = PortfolioGuard(
        GuardConfig(
            total_cap_usdt=total_cap_usdt,
            per_symbol_cap_usdt=per_symbol_cap_usdt,
            venue=venue,
            soft_cap_pct=soft_cap_pct,
            soft_cap_grace_sec=soft_cap_grace_sec,
        )
    )
    dguard = DailyGuard(
        GuardLimits(
            daily_max_loss_pct=daily_max_loss_pct,
            daily_max_drawdown_pct=daily_max_drawdown_pct,
            halt_action="close_all",
        ),
        venue=venue,
    )
    corr = CorrelationService()
    risk = RiskService(risk_core, guard, dguard, corr_service=corr)
    broker = PaperAdapter(fee_bps=1.5)
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
        if dry_run:
            resp = await broker.place_order(cfg.symbol, side, "market", qty)
        else:
            resp = await exec_adapter.place_order(
                cfg.symbol, side, "market", qty, mark_price=closed.c
            )
        log.info("LIVE FILL %s", resp)
        risk.on_fill(cfg.symbol, side, qty, venue=venue if not dry_run else "paper")


async def run_live_real(
    exchange: str = "binance",
    market: str = "spot",
    symbols: List[str] | None = None,
    trade_qty: float = 0.001,
    leverage: int = 1,
    dry_run: bool = False,
    *,
    i_know_what_im_doing: bool,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    daily_max_loss_pct: float = 0.05,
    daily_max_drawdown_pct: float = 0.05,
    corr_threshold: float = 0.8,
    config_path: str | None = None,
) -> None:
    """Run a simple live loop on a real crypto exchange."""

    if not i_know_what_im_doing:
        raise ValueError("Real trading requires --i-know-what-im-doing flag")
    key, secret = _get_keys(exchange)
    if not key or not secret:
        raise RuntimeError(f"Missing API keys for {exchange} in .env")
    if (exchange, market) not in ADAPTERS:
        raise ValueError(f"Unsupported combination {exchange} {market}")
    symbols = symbols or ["BTC/USDT"]
    cfgs = [
        _SymbolConfig(symbol=s.upper().replace("-", "/"), trade_qty=trade_qty)
        for s in symbols
    ]
    tasks = [
        _run_symbol(
            exchange,
            market,
            c,
            leverage,
            dry_run,
            total_cap_usdt,
            per_symbol_cap_usdt,
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


__all__ = ["run_live_real"]

