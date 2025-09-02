from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Type, List, Any
import time

import pandas as pd

from .runner import BarAggregator
from ..config import settings
from ..config.hydra_conf import load_config
from ..strategies import STRATEGIES
from ..strategies.breakout_atr import BreakoutATR
from ..risk.service import load_positions
from ..risk.daily_guard import DailyGuard, GuardLimits
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.correlation_service import CorrelationService
from ..risk.service import RiskService
from ..execution.paper import PaperAdapter
from ..broker.broker import Broker

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
    risk_pct: float

async def _run_symbol(
    exchange: str,
    market: str,
    cfg: _SymbolConfig,
    leverage: int,
    dry_run: bool,
    total_cap_pct: float,
    per_symbol_cap_pct: float,
    soft_cap_pct: float,
    soft_cap_grace_sec: int,
    daily_max_loss_pct: float,
    daily_max_drawdown_pct: float,
    corr_threshold: float,
    strategy_name: str,
    params: dict | None = None,
    config_path: str | None = None,
) -> None:
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
    cfg_app = load_config()
    tick_size = float(cfg_app.exchange_configs.get(venue, {}).get("tick_size", 0.0))
    agg = BarAggregator()
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    params = params or {}
    strat = strat_cls(config_path=config_path, **params) if (config_path or params) else strat_cls()
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
    broker = PaperAdapter(fee_bps=1.5)
    exec_broker = Broker(exec_adapter if not dry_run else broker)
    risk = RiskService(
        guard,
        dguard,
        corr_service=corr,
        account=broker.account,
        risk_pct=cfg.risk_pct,
    )
    risk.allow_short = market != "spot"
    strat.risk_service = risk
    limit_offset = settings.limit_offset_ticks * tick_size
    tif = f"GTD:{settings.limit_expiry_sec}|PO"
    try:
        guard.refresh_usd_caps(broker.equity({}))
    except Exception:
        guard.refresh_usd_caps(0.0)
    engine = get_engine() if _CAN_PG else None
    if engine is not None:
        pos_map = load_positions(engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(
                guard.cfg.venue, sym, data.get("qty", 0.0), entry_price=data.get("avg_price")
            )

    async for t in ws.stream_trades(cfg.symbol):
        ts: datetime = t.get("ts") or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t.get("qty") or 0.0)
        broker.update_last_price(cfg.symbol, px)
        risk.mark_price(cfg.symbol, px)
        risk.update_correlation(corr._returns.corr(), corr_threshold)
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
        bar = {"window": df, "symbol": cfg.symbol}
        sig = strat.on_bar(bar)
        if sig is None:
            continue
        signal_ts = getattr(sig, "signal_ts", time.time())
        allowed, reason, delta = risk.check_order(
            cfg.symbol,
            sig.side,
            closed.c,
            strength=sig.strength,
            corr_threshold=corr_threshold,
            volatility=bar.get("atr") or bar.get("volatility"),
            target_volatility=bar.get("target_volatility"),
        )
        if not allowed or abs(delta) <= 0:
            if reason:
                log.warning("[PG] Bloqueado %s: %s", cfg.symbol, reason)
            continue
        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)
        price = (
            sig.limit_price
            if sig.limit_price is not None
            else (closed.c - limit_offset if side == "buy" else closed.c + limit_offset)
        )
        notional = qty * price
        if not risk.register_order(cfg.symbol, notional):
            continue
        prev_rpnl = broker.state.realized_pnl
        resp = await exec_broker.place_limit(
            cfg.symbol,
            side,
            price,
            qty,
            tif=tif,
            on_partial_fill=lambda *_: "re_quote",
            on_order_expiry=lambda *_: "re_quote",
            signal_ts=signal_ts,
        )
        log.info("LIVE FILL %s", resp)
        filled_qty = float(resp.get("filled_qty", 0.0))
        pending_qty = float(resp.get("pending_qty", 0.0))
        risk.account.update_open_order(cfg.symbol, side, filled_qty + pending_qty)
        risk.on_fill(
            cfg.symbol, side, filled_qty, venue=venue if not dry_run else "paper"
        )
        delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
        halted, reason = risk.daily_mark(broker, cfg.symbol, px, delta_rpnl)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break

async def run_live_testnet(
    exchange: str = "binance",
    market: str = "spot",
    symbols: List[str] | None = None,
    risk_pct: float = 0.0,
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
    strategy_name: str = "breakout_atr",
    config_path: str | None = None,
    params: dict | None = None,
) -> None:
    """Run a simple live loop on a crypto exchange testnet."""
    if (exchange, market) not in ADAPTERS:
        raise ValueError(f"Unsupported combination {exchange} {market}")
    symbols = symbols or ["BTC/USDT"]
    cfgs = [
        _SymbolConfig(symbol=s.upper().replace("-", "/"), risk_pct=risk_pct)
        for s in symbols
    ]
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
            strategy_name=strategy_name,
            params=params,
            config_path=config_path,
        )
        for c in cfgs
    ]
    await asyncio.gather(*tasks)
