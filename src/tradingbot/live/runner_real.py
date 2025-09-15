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
import errno
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple
import time
import json

import pandas as pd
import uvicorn
import ccxt

from sqlalchemy.exc import OperationalError
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
from ..adapters.binance import BinanceWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..adapters.bybit_spot import BybitSpotAdapter as BybitSpotWSAdapter, BybitSpotAdapter
from ..adapters.okx_spot import OKXSpotAdapter as OKXSpotWSAdapter, OKXSpotAdapter
from ..adapters.bybit_futures import BybitFuturesAdapter
from ..adapters.okx_futures import OKXFuturesAdapter
from monitoring import panel
from ..execution.order_sizer import adjust_qty
from ..core.symbols import normalize
from ..utils.price import limit_price_from_close

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
    try:
        task = asyncio.create_task(server.serve())
    except SystemExit as exc:  # pragma: no cover - defensive
        raise OSError(errno.EADDRINUSE) from exc
    while not server.started and not task.done():
        await asyncio.sleep(0.1)
    if task.done():
        exc = task.exception()
        if exc is not None:
            if isinstance(exc, SystemExit):
                raise OSError(errno.EADDRINUSE) from exc
            raise exc
    return server


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
    risk_pct: float


async def _run_symbol(
    exchange: str,
    market: str,
    cfg: _SymbolConfig,
    leverage: int,
    dry_run: bool,
    total_cap_pct: float | None,
    per_symbol_cap_pct: float | None,
    soft_cap_pct: float,
    soft_cap_grace_sec: int,
    daily_max_loss_pct: float,
    daily_max_drawdown_pct: float,
    corr_threshold: float,
    strategy_name: str,
    params: dict | None = None,
    config_path: str | None = None,
    timeframe: str = "1m",
    risk_per_trade: float = 1.0,
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
    slippage_bps: float = 0.0,
    reprice_bps: float = 0.0,
    min_notional: float = 0.0,
    step_size: float = 0.0,
) -> None:
    ws_cls, exec_cls, venue = ADAPTERS[(exchange, market)]
    raw_symbol = cfg.symbol
    symbol = normalize(cfg.symbol)
    timeframe_seconds = ccxt.Exchange.parse_timeframe(timeframe)
    expiry = timeframe_seconds
    log.info("Connecting to %s %s for %s", exchange, market, symbol)
    api_key, api_secret = _get_keys(exchange)
    exec_kwargs: Dict[str, Any] = {"api_key": api_key, "api_secret": api_secret}
    if market == "futures":
        exec_kwargs["leverage"] = leverage
    ws = ws_cls()
    exec_adapter = exec_cls(**exec_kwargs)
    cfg_app = load_config()
    tick_size = 0.0
    meta = getattr(exec_adapter, "meta", None)
    if meta is not None:
        try:
            fetch_symbol = None
            symbols = getattr(meta.client, "symbols", [])
            if symbols:
                fetch_symbol = next((s for s in symbols if normalize(s) == symbol), None)
            if fetch_symbol is None:
                fetch_symbol = raw_symbol.replace("-", "/")
            rules = meta.rules_for(fetch_symbol)
            tick_size = float(getattr(rules, "price_step", 0.0) or 0.0)
        except Exception:
            tick_size = 0.0
    agg = BarAggregator(timeframe=timeframe)
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    params = params or {}
    strat = strat_cls(config_path=config_path, **params) if (config_path or params) else strat_cls()
    guard = PortfolioGuard(
        GuardConfig(
            total_cap_pct=total_cap_pct,
            per_symbol_cap_pct=per_symbol_cap_pct,
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
    import inspect
    broker_kwargs = {
        "maker_fee_bps": maker_fee_bps,
        "taker_fee_bps": taker_fee_bps,
        "min_notional": min_notional,
        "step_size": step_size,
        "slip_bps_per_qty": slip_bps_per_qty,
    }
    sig = inspect.signature(PaperAdapter)
    broker = PaperAdapter(
        **{k: v for k, v in broker_kwargs.items() if k in sig.parameters}
    )
    broker.account.market_type = market
    exec_broker = Broker(exec_adapter if not dry_run else broker)
    tif = f"GTD:{expiry}|PO"
    risk = RiskService(
        guard,
        dguard,
        corr_service=corr,
        account=broker.account,
        risk_pct=cfg.risk_pct,
        risk_per_trade=risk_per_trade,
        market_type=market,
    )
    strat.risk_service = risk

    def on_order_cancel(order, res: dict) -> None:
        """Track broker order cancellations."""
        symbol = res.get("symbol") or getattr(order, "symbol", None)
        side = res.get("side") or getattr(order, "side", None)
        pending_qty = res.get("pending_qty")
        if pending_qty is None and order is not None:
            pending_qty = getattr(order, "pending_qty", None)
        if pending_qty is None:
            pending_qty = res.get("qty", 0.0)
        pending_qty = float(pending_qty or 0.0)
        if symbol and side and pending_qty > 0:
            risk.account.update_open_order(symbol, side, -pending_qty)
        locked = risk.account.get_locked_usd(symbol) if symbol else 0.0
        if not risk.account.open_orders.get(symbol):
            locked = 0.0
        log.info(
            "METRICS %s",
            json.dumps(
                {"event": "cancel", "reason": res.get("reason"), "locked": locked}
            ),
        )
        already_completed = order is not None and getattr(
            order, "_risk_order_completed", False
        )
        if not already_completed:
            risk.complete_order()
            if order is not None:
                setattr(order, "_risk_order_completed", True)

    last_price = 0.0

    def _wrap_cb(orig_cb, *, call_cancel=False):
        def _cb(order, res):
            if call_cancel:
                on_order_cancel(order, res)
            action = orig_cb(order, res) if orig_cb else None
            if not call_cancel:
                pending_raw = res.get("pending_qty")
                pending_qty = None
                if pending_raw is not None:
                    try:
                        pending_qty = float(pending_raw)
                    except (TypeError, ValueError):
                        pending_qty = None
                if pending_qty is not None and pending_qty <= 0:
                    already_completed = order is not None and getattr(
                        order, "_risk_order_completed", False
                    )
                    if not already_completed:
                        risk.complete_order()
                        if order is not None:
                            setattr(order, "_risk_order_completed", True)
            if action in {"re_quote", "requote", "re-quote"}:
                return None
            return action
        return _cb

    on_pf = _wrap_cb(getattr(strat, "on_partial_fill", None))
    on_oe = _wrap_cb(getattr(strat, "on_order_expiry", None), call_cancel=True)
    try:
        guard.refresh_usd_caps(broker.equity({}))
    except Exception:
        guard.refresh_usd_caps(0.0)
    engine = None
    if _CAN_PG:
        while True:
            try:
                engine = get_engine()
                break
            except OperationalError:
                log.warning("QuestDB no disponible, reintentando en 5s")
                await asyncio.sleep(5)
    if engine is not None:
        pos_map = load_positions(engine, guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(
                guard.cfg.venue, sym, data.get("qty", 0.0), entry_price=data.get("avg_price")
            )

    async for t in ws.stream_trades(symbol):
        ts: datetime = t.get("ts") or datetime.now(timezone.utc)
        px: float = float(t["price"])
        qty: float = float(t.get("qty") or 0.0)
        broker.update_last_price(symbol, px)
        risk.mark_price(symbol, px)
        risk.update_correlation(corr.get_correlations(), corr_threshold)
        last_price = px
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
                last_close = agg.completed[-1].c if agg.completed else px
                price = limit_price_from_close(close_side, last_close, tick_size)
                prev_rpnl = broker.state.realized_pnl
                qty_close = adjust_qty(
                    abs(pos_qty), price, min_notional, step_size, risk.min_order_qty
                )
                if qty_close <= 0:
                    log.info(
                        "Skipping order: qty %.8f below min threshold", abs(pos_qty)
                    )
                    log.info(
                        "METRICS %s",
                        json.dumps({"event": "skip", "reason": "below_min_qty"}),
                    )
                    continue
                resp = await exec_broker.place_limit(
                    symbol,
                    close_side,
                    price,
                    qty_close,
                    tif=tif,
                    on_partial_fill=on_pf,
                    on_order_expiry=on_oe,
                    slip_bps=slippage_bps,
                )
                filled_qty = float(resp.get("filled_qty", 0.0))
                pending_qty = float(resp.get("pending_qty", 0.0))
                prev_pending = risk.account.open_orders.get(symbol, {}).get(
                    close_side, 0.0
                )
                delta_open = (
                    filled_qty + pending_qty - prev_pending
                    if not dry_run
                    else pending_qty - prev_pending
                )
                risk.account.update_open_order(symbol, close_side, delta_open)
                risk.on_fill(
                    symbol,
                    close_side,
                    filled_qty,
                    venue=venue if not dry_run else "paper",
                )
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
                    side = trade["side"] if delta_qty > 0 else (
                        "sell" if trade["side"] == "buy" else "buy"
                    )
                    last_close = agg.completed[-1].c if agg.completed else px
                    price = limit_price_from_close(side, last_close, tick_size)
                    qty_scale = adjust_qty(
                        abs(delta_qty), price, min_notional, step_size, risk.min_order_qty
                    )
                    if qty_scale <= 0:
                        log.info(
                            "Skipping order: qty %.8f below min threshold", abs(delta_qty)
                        )
                        log.info(
                            "METRICS %s",
                            json.dumps({"event": "skip", "reason": "below_min_qty"}),
                        )
                        continue
                    prev_rpnl = broker.state.realized_pnl
                    resp = await exec_broker.place_limit(
                        symbol,
                        side,
                        price,
                        qty_scale,
                        tif=tif,
                        on_partial_fill=on_pf,
                        on_order_expiry=on_oe,
                        slip_bps=slippage_bps,
                    )
                    filled_qty = float(resp.get("filled_qty", 0.0))
                    pending_qty = float(resp.get("pending_qty", 0.0))
                    prev_pending = risk.account.open_orders.get(symbol, {}).get(
                        side, 0.0
                    )
                    delta_open = (
                        filled_qty + pending_qty - prev_pending
                        if not dry_run
                        else pending_qty - prev_pending
                    )
                    risk.account.update_open_order(symbol, side, delta_open)
                    risk.on_fill(
                        symbol,
                        side,
                        filled_qty,
                        venue=venue if not dry_run else "paper",
                    )
                    delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                    halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                    if halted:
                        log.error("[HALT] motivo=%s", reason)
                        break
                continue
        closed = agg.on_trade(ts, px, qty)
        if closed is None:
            continue
        df: pd.DataFrame = agg.last_n_bars_df(200)
        if len(df) < 140:
            continue
        bar = {"window": df, "symbol": symbol}
        sig = strat.on_bar(bar)
        if sig is None:
            continue
        if (
            sig.side == "sell"
            and not risk.allow_short
            and risk.account.current_exposure(symbol)[0] <= 0
        ):
            log.warning("[PG] Bloqueado %s: short_not_allowed", symbol)
            log.info(
                "METRICS %s",
                json.dumps({"event": "skip", "reason": "short_not_allowed"}),
            )
            continue
        signal_ts = getattr(sig, "signal_ts", time.time())
        allowed, reason, delta = risk.check_order(
            symbol,
            sig.side,
            closed.c,
            strength=sig.strength,
            volatility=bar.get("atr") or bar.get("volatility"),
            target_volatility=bar.get("target_volatility"),
        )
        if not allowed:
            if reason == "below_min_qty":
                log.info(
                    "Skipping order: qty %.8f below min threshold", abs(delta)
                )
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "skip", "reason": "below_min_qty"}),
                )
            elif reason:
                log.warning("[PG] Bloqueado %s: %s", symbol, reason)
            continue
        if abs(delta) <= 0:
            continue
        side = "buy" if delta > 0 else "sell"
        price = (
            sig.limit_price
            if sig.limit_price is not None
            else limit_price_from_close(side, closed.c, tick_size)
        )
        qty = adjust_qty(abs(delta), price, min_notional, step_size, risk.min_order_qty)
        if qty <= 0:
            log.info(
                "Skipping order: qty %.8f below min threshold", abs(delta)
            )
            log.info(
                "METRICS %s",
                json.dumps({"event": "skip", "reason": "below_min_qty"}),
            )
            continue
        notional = qty * price
        if not risk.register_order(symbol, notional):
            continue
        prev_rpnl = broker.state.realized_pnl
        resp = await exec_broker.place_limit(
            symbol,
            side,
            price,
            qty,
            tif=tif,
            on_partial_fill=on_pf,
            on_order_expiry=on_oe,
            signal_ts=signal_ts,
            slip_bps=slippage_bps,
        )
        log.info("LIVE FILL %s", resp)
        filled_qty = float(resp.get("filled_qty", 0.0))
        pending_qty = float(resp.get("pending_qty", 0.0))
        prev_pending = risk.account.open_orders.get(symbol, {}).get(side, 0.0)
        delta_open = (
            filled_qty + pending_qty - prev_pending
            if not dry_run
            else pending_qty - prev_pending
        )
        risk.account.update_open_order(symbol, side, delta_open)
        risk.on_fill(
            symbol, side, filled_qty, venue=venue if not dry_run else "paper"
        )
        delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
        halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break


async def run_live_real(
    exchange: str = "binance",
    market: str = "spot",
    symbols: List[str] | None = None,
    risk_pct: float = 0.0,
    leverage: int = 1,
    dry_run: bool = False,
    *,
    i_know_what_im_doing: bool,
    metrics_port: int = 8000,
    total_cap_pct: float | None = None,
    per_symbol_cap_pct: float | None = None,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
    daily_max_loss_pct: float = 0.05,
    daily_max_drawdown_pct: float = 0.05,
    corr_threshold: float = 0.8,
    strategy_name: str = "breakout_atr",
    config_path: str | None = None,
    params: dict | None = None,
    timeframe: str = "1m",
    risk_per_trade: float = 1.0,
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
    slippage_bps: float = 0.0,
    slip_bps_per_qty: float = 0.0,
    reprice_bps: float = 0.0,
    min_notional: float = 0.0,
    step_size: float = 0.0,
) -> None:
    """Run a simple live loop on a real crypto exchange.

    Parameters
    ----------
    slip_bps_per_qty:
        Optional manual slippage in basis points applied per unit of traded
        quantity. When omitted, slippage is estimated automatically from order
        book depth or historical fills during dry runs.
    """
    log.info("Starting real runner for %s %s", exchange, market)

    if not i_know_what_im_doing:
        raise ValueError("Real trading requires --i-know-what-im-doing flag")
    key, secret = _get_keys(exchange)
    if not key or not secret:
        raise RuntimeError(f"Missing API keys for {exchange} in .env")
    if (exchange, market) not in ADAPTERS:
        raise ValueError(f"Unsupported combination {exchange} {market}")
    symbols = symbols or ["BTC/USDT"]
    cfgs = [
        _SymbolConfig(symbol=s.upper().replace("-", "/"), risk_pct=risk_pct)
        for s in symbols
    ]
    port = metrics_port
    while True:
        try:
            server = await _start_metrics(port)
            if port == 0 and server.servers:
                actual = server.servers[0].sockets[0].getsockname()[1]
                log.info("metrics server listening on port %s", actual)
            break
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EADDRINUSE:
                log.warning("metrics port %s in use, trying %s", port, port + 1)
                port += 1
                continue
            raise
        except SystemExit as exc:  # pragma: no cover - defensive
            log.warning("metrics port %s in use, trying %s", port, port + 1)
            port += 1
            continue
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
            timeframe=timeframe,
            risk_per_trade=risk_per_trade,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            slip_bps_per_qty=slip_bps_per_qty,
            reprice_bps=reprice_bps,
            min_notional=min_notional,
            step_size=step_size,
        )
        for c in cfgs
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        server.should_exit = True


__all__ = ["run_live_real"]

