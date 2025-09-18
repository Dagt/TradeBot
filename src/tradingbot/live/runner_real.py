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
from ._symbol_rules import resolve_symbol_rules
from ..utils.metrics import CANCELS
from ..utils.price import limit_price_from_close
from ._metrics import infer_maker_flag

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
    min_qty_val = 0.0
    meta = getattr(exec_adapter, "meta", None)
    if meta is not None:
        try:
            rules, _ = resolve_symbol_rules(meta, raw_symbol, symbol)
            step_candidate = float(getattr(rules, "qty_step", 0.0) or 0.0)
            if step_size <= 0 and step_candidate > 0:
                step_size = step_candidate
            if min_notional <= 0:
                try:
                    min_notional = float(getattr(rules, "min_notional", 0.0) or 0.0)
                except (TypeError, ValueError):
                    min_notional = 0.0
            try:
                min_qty_val = float(getattr(rules, "min_qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                min_qty_val = 0.0
            tick_size = float(getattr(rules, "price_step", 0.0) or 0.0)
        except Exception:
            tick_size = 0.0
    if step_size <= 0:
        step_size = 1e-9
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
    min_qty_value = min_qty_val if min_qty_val > 0 else 0.0
    step_value = step_size if step_size > 0 else 0.0
    min_order_qty = max(min_qty_value, step_value)
    risk.min_order_qty = min_order_qty if min_order_qty > 0 else 1e-9
    risk.min_notional = float(min_notional if min_notional > 0 else 0.0)
    strat.risk_service = risk

    trades_closed = 0
    trades_won = 0
    pnl_won_total = 0.0
    pnl_lost_total = 0.0
    try:
        total_pnl = float(getattr(broker.state, "realized_pnl", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_pnl = 0.0

    def _flat_threshold() -> float:
        base = risk.min_order_qty
        if base > 0:
            return base
        step = step_size if step_size > 0 else 0.0
        if step > 0:
            return step
        return 1e-9

    def _position_closed(before: float, after: float) -> bool:
        threshold = _flat_threshold()
        if abs(after) <= threshold:
            return True
        return (before > 0 > after) or (before < 0 < after)

    def _record_trade(delta_pnl: float) -> None:
        nonlocal trades_closed, trades_won, pnl_won_total, pnl_lost_total, total_pnl
        total_pnl += float(delta_pnl)
        trades_closed += 1
        if delta_pnl > 1e-9:
            trades_won += 1
            pnl_won_total += float(delta_pnl)
        elif delta_pnl < -1e-9:
            pnl_lost_total += float(delta_pnl)
        losses = trades_closed - trades_won
        expectancy = total_pnl / trades_closed if trades_closed else 0.0
        avg_win = pnl_won_total / trades_won if trades_won else 0.0
        avg_loss = abs(pnl_lost_total / losses) if losses else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss else 0.0
        hit_rate = (trades_won / trades_closed) * 100 if trades_closed else 0.0
        payload = {
            "event": "trade",
            "pnl": float(delta_pnl),
            "trade_pnl": float(delta_pnl),
            "trades_closed": trades_closed,
            "trades_won": trades_won,
            "pnl_won": pnl_won_total,
            "pnl_lost": pnl_lost_total,
            "expectancy": expectancy,
            "payoff_ratio": payoff_ratio,
            "hit_rate": hit_rate,
            "hit%": hit_rate,
        }
        log.info("METRICS %s", json.dumps(payload))
        log.info("METRICS %s", json.dumps({"pnl": total_pnl}))

    def _recalc_locked_total() -> float:
        """Recalculate total notional locked across all open orders."""

        account = getattr(risk, "account", None)
        if account is None:
            return 0.0

        open_orders = getattr(account, "open_orders", None)
        if not isinstance(open_orders, dict) or not open_orders:
            setattr(account, "locked_total", 0.0)
            setattr(risk, "locked_total", 0.0)
            return 0.0

        prices = getattr(account, "prices", {})
        get_locked = getattr(account, "get_locked_usd", None)

        def _symbol_locked(sym: str, orders: object) -> float:
            if callable(get_locked):
                try:
                    return float(get_locked(sym))
                except Exception:  # pragma: no cover - fallback below
                    pass

            total_qty = 0.0
            if isinstance(orders, dict):
                for _, qty_raw in orders.items():
                    try:
                        total_qty += abs(float(qty_raw))
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    total_qty = abs(float(orders))
                except (TypeError, ValueError):
                    total_qty = 0.0

            try:
                price = float(prices.get(sym, 0.0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            return total_qty * price

        total_locked = 0.0
        for sym, orders in list(open_orders.items()):
            total_locked += _symbol_locked(sym, orders)

        if total_locked <= 1e-9:
            total_locked = 0.0

        setattr(account, "locked_total", total_locked)
        setattr(risk, "locked_total", total_locked)
        return total_locked

    def on_order_cancel(order, res: dict) -> None:
        """Track broker order cancellations."""
        if not isinstance(res, dict):
            return
        status = str(res.get("status", "")).lower()
        if status not in {"canceled", "cancelled", "expired"}:
            return
        if res.get("_cancel_handled"):
            return
        res["_cancel_handled"] = True
        CANCELS.inc()
        symbol = res.get("symbol") or getattr(order, "symbol", None)
        side = res.get("side") or getattr(order, "side", None)
        side_norm = str(side).lower() if isinstance(side, str) else None
        lookup_side = side_norm or side
        pending_raw = res.get("pending_qty")
        if pending_raw is None and order is not None:
            pending_raw = getattr(order, "pending_qty", None)
        if pending_raw is None:
            pending_raw = res.get("qty")
        pending_qty = None
        if pending_raw is not None:
            try:
                pending_qty = float(pending_raw)
            except (TypeError, ValueError):
                pending_qty = None
        prev_pending = 0.0
        if symbol and lookup_side:
            try:
                prev_pending = float(
                    risk.account.open_orders.get(symbol, {}).get(lookup_side, 0.0)
                    or 0.0
                )
            except (TypeError, ValueError):
                prev_pending = 0.0
        if (pending_qty is None or pending_qty == 0.0) and symbol and lookup_side:
            pending_qty = prev_pending
        filled_qty = 0.0
        try:
            filled_qty = float(res.get("filled_qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            filled_qty = 0.0
        metric_pending_override: float | None = None
        if filled_qty > 0:
            if symbol and lookup_side:
                delta_pending = -prev_pending
                risk.account.update_open_order(symbol, lookup_side, delta_pending)
            metric_pending_override = 0.0
        elif symbol and lookup_side and pending_qty and pending_qty > 0:
            risk.account.update_open_order(symbol, lookup_side, -pending_qty)
        metric_pending = res.get("pending_qty", pending_qty)
        if metric_pending_override is not None:
            metric_pending = metric_pending_override
        try:
            metric_pending_val = float(metric_pending)
        except (TypeError, ValueError):
            metric_pending_val = 0.0
        if metric_pending_val <= 0:
            risk.complete_order()
            if order is not None:
                setattr(order, "_risk_order_completed", True)
            locked_total = _recalc_locked_total()
            log.info(
                "METRICS %s",
                json.dumps(
                    {"event": "cancel", "reason": res.get("reason"), "locked": locked_total}
                ),
            )
            return  # treat as filled; no cancel handling needed
        already_completed = order is not None and getattr(
            order, "_risk_order_completed", False
        )
        if not already_completed:
            risk.complete_order()
            if order is not None:
                setattr(order, "_risk_order_completed", True)
        locked_total = _recalc_locked_total()
        log.info(
            "METRICS %s",
            json.dumps(
                {"event": "cancel", "reason": res.get("reason"), "locked": locked_total}
            ),
        )

    last_price = 0.0

    def _wrap_cb(orig_cb, *, call_cancel=False):
        def _cb(order, res):
            status = ""
            if isinstance(res, dict):
                status = str(res.get("status", "")).lower()
            res_dict = res if isinstance(res, dict) else {}
            if call_cancel:
                if order is not None:
                    res_dict.setdefault("symbol", getattr(order, "symbol", None))
                    res_dict.setdefault("side", getattr(order, "side", None))
                    res_dict.setdefault("pending_qty", getattr(order, "pending_qty", None))
                res = res_dict or res
                if not res_dict.get("_cancel_handled"):
                    on_order_cancel(order, res_dict)
            else:
                res = res_dict or res
            action = orig_cb(order, res) if orig_cb else None
            filled_qty = 0.0
            if isinstance(res, dict):
                try:
                    filled_qty = float(res.get("filled_qty", 0.0) or 0.0)
                except (TypeError, ValueError):
                    filled_qty = 0.0
            if filled_qty > 0:
                symbol = None
                side = None
                if order is not None:
                    symbol = getattr(order, "symbol", None)
                    side = getattr(order, "side", None)
                if symbol is None and isinstance(res, dict):
                    symbol = res.get("symbol")
                if side is None and isinstance(res, dict):
                    side = res.get("side")
                price_raw = None
                if isinstance(res, dict):
                    price_raw = res.get("price") or res.get("avg_price")
                if price_raw is None and order is not None:
                    price_raw = getattr(order, "price", None)
                exec_price = None
                if price_raw is not None:
                    try:
                        exec_price = float(price_raw)
                    except (TypeError, ValueError):
                        exec_price = None
                base_price = getattr(order, "price", None) if order is not None else None
                slippage_bps = None
                if isinstance(res, dict):
                    slippage_bps = res.get("slippage_bps")
                if slippage_bps is None and exec_price is not None and base_price:
                    try:
                        base_price_f = float(base_price)
                        if base_price_f:
                            slippage_bps = ((exec_price - base_price_f) / base_price_f) * 10000.0
                    except (TypeError, ValueError, ZeroDivisionError):
                        slippage_bps = 0.0
                fee = None
                fee_bps = None
                if isinstance(res, dict):
                    fee = res.get("fee")
                    fee_bps = res.get("fee_bps")
                if fee is None and exec_price is not None:
                    fee_type = (res.get("fee_type") if isinstance(res, dict) else None) or ""
                    fee_type = str(fee_type).lower()
                    if fee_bps is None:
                        if fee_type == "maker":
                            fee_bps = getattr(exec_broker, "maker_fee_bps", getattr(broker, "maker_fee_bps", 0.0))
                        elif fee_type == "taker":
                            fee_bps = getattr(exec_broker, "taker_fee_bps", getattr(broker, "taker_fee_bps", 0.0))
                        else:
                            fee_bps = getattr(exec_broker, "maker_fee_bps", getattr(broker, "maker_fee_bps", 0.0))
                    try:
                        fee = filled_qty * exec_price * (float(fee_bps) / 10000.0)
                    except (TypeError, ValueError):
                        fee = 0.0
                side_norm = str(side).lower() if side is not None else None
                maker_flag = infer_maker_flag(
                    res if isinstance(res, dict) else None,
                    exec_price,
                    base_price,
                )
                log.info(
                    "METRICS %s",
                    json.dumps(
                        {
                            "event": "fill",
                            "side": side,
                            "price": exec_price,
                            "qty": filled_qty,
                            "fee": 0.0 if fee is None else fee,
                            "slippage_bps": (
                                float(slippage_bps)
                                if slippage_bps is not None
                                else 0.0
                            ),
                            "maker": maker_flag,
                        }
                    ),
                )
                if symbol and side_norm:
                    pending_qty = None
                    if isinstance(res, dict):
                        pending_raw = res.get("pending_qty")
                        if pending_raw is not None:
                            try:
                                pending_qty = float(pending_raw)
                            except (TypeError, ValueError):
                                pending_qty = None
                    account_open_orders = getattr(risk.account, "open_orders", None)
                    update_open = getattr(risk.account, "update_open_order", None)
                    prev_pending = 0.0
                    if isinstance(account_open_orders, dict):
                        prev_pending = float(
                            account_open_orders.get(symbol, {}).get(side_norm, 0.0) or 0.0
                        )
                    if pending_qty is not None and callable(update_open):
                        if isinstance(account_open_orders, dict):
                            delta_pending = pending_qty - prev_pending
                            if abs(delta_pending) > 1e-12:
                                update_open(symbol, side_norm, delta_pending)
                        else:
                            update_open(symbol, side_norm, pending_qty)
                    target_qty = None
                    if isinstance(res, dict) and res.get("pos_qty") is not None:
                        try:
                            target_qty = float(res.get("pos_qty"))
                        except (TypeError, ValueError):
                            target_qty = None
                    positions = getattr(risk.account, "positions", {})
                    current_qty = 0.0
                    if isinstance(positions, dict):
                        current_qty = float(positions.get(symbol, 0.0) or 0.0)
                    if target_qty is None:
                        direction = 1.0 if side_norm == "buy" else -1.0
                        target_qty = current_qty + direction * filled_qty
                    delta_qty = target_qty - current_qty
                    price_for_position = exec_price
                    if price_for_position is None and base_price is not None:
                        try:
                            price_for_position = float(base_price)
                        except (TypeError, ValueError):
                            price_for_position = None
                    update_position = getattr(risk.account, "update_position", None)
                    if abs(delta_qty) > 1e-12 and callable(update_position):
                        update_position(symbol, delta_qty, price=price_for_position)
                    current_exposure_fn = getattr(risk.account, "current_exposure", None)
                    exposure_qty = target_qty
                    if callable(current_exposure_fn):
                        try:
                            exposure_qty = float(current_exposure_fn(symbol)[0])
                        except Exception:
                            exposure_qty = float(target_qty)
                    locked = _recalc_locked_total()
                    log.info(
                        "METRICS %s",
                        json.dumps({"exposure": exposure_qty, "locked": locked}),
                    )
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
                    locked_after_completion = _recalc_locked_total()
                    symbol_for_metrics = None
                    if order is not None:
                        symbol_for_metrics = getattr(order, "symbol", None)
                    if symbol_for_metrics is None and isinstance(res, dict):
                        symbol_for_metrics = res.get("symbol")
                    exposure_after = 0.0
                    if symbol_for_metrics:
                        try:
                            exposure_after = float(
                                risk.account.current_exposure(symbol_for_metrics)[0]
                            )
                        except Exception:
                            exposure_after = 0.0
                    log.info(
                        "METRICS %s",
                        json.dumps(
                            {"exposure": exposure_after, "locked": locked_after_completion}
                        ),
                    )
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
        threshold = _flat_threshold()
        if trade and abs(pos_qty) > threshold:
            risk.update_trailing(trade, px)
            trade["_trail_done"] = True
            decision = risk.manage_position(trade)
            if decision == "close":
                close_side = "sell" if pos_qty > 0 else "buy"
                prev_pos_qty = pos_qty
                last_close = agg.completed[-1].c if agg.completed else px
                price = limit_price_from_close(close_side, last_close, tick_size)
                try:
                    prev_rpnl = float(getattr(broker.state, "realized_pnl", 0.0))
                except (TypeError, ValueError):
                    prev_rpnl = total_pnl
                qty_close = adjust_qty(
                    abs(pos_qty), price, min_notional, step_size, risk.min_order_qty
                )
                if qty_close < threshold:
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
                status = str(resp.get("status", ""))
                filled_qty = float(resp.get("filled_qty", 0.0))
                pending_qty = float(resp.get("pending_qty", 0.0))
                if status == "rejected":
                    continue
                log_order = False
                order_qty = qty_close
                if status in {"open", "filled"}:
                    log_order = True
                elif status == "canceled" and filled_qty > 0:
                    log_order = True
                    order_qty = filled_qty
                if log_order:
                    log.info(
                        "METRICS %s",
                        json.dumps(
                            {
                                "event": "order",
                                "side": close_side,
                                "price": price,
                                "qty": order_qty,
                                "fee": 0.0,
                                "pnl": broker.state.realized_pnl,
                            }
                        ),
                    )
                    risk.account.update_open_order(symbol, close_side, pending_qty)
                    cur_qty = risk.account.current_exposure(symbol)[0]
                    locked = _recalc_locked_total()
                    log.info(
                        "METRICS %s",
                        json.dumps({"exposure": cur_qty, "locked": locked}),
                    )
                risk.on_fill(
                    symbol,
                    close_side,
                    filled_qty,
                    venue=venue if not dry_run else "paper",
                )
                realized_raw = resp.get("realized_pnl", getattr(broker.state, "realized_pnl", prev_rpnl))
                try:
                    realized_val = float(realized_raw)
                except (TypeError, ValueError):
                    realized_val = prev_rpnl + 0.0
                delta_rpnl = realized_val - prev_rpnl
                try:
                    post_qty = float(risk.account.current_exposure(symbol)[0])
                except Exception:
                    post_qty = pos_qty
                if filled_qty > 0 and _position_closed(prev_pos_qty, post_qty):
                    _record_trade(delta_rpnl)
                halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                if halted:
                    log.error("[HALT] motivo=%s", reason)
                    break
                continue
            if decision in {"scale_in", "scale_out"}:
                target = risk.calc_position_size(trade.get("strength", 1.0), px, clamp=False)
                delta_qty = target - abs(pos_qty)
                if abs(delta_qty) > threshold:
                    side = trade["side"] if delta_qty > 0 else (
                        "sell" if trade["side"] == "buy" else "buy"
                    )
                    last_close = agg.completed[-1].c if agg.completed else px
                    price = limit_price_from_close(side, last_close, tick_size)
                    qty_scale = adjust_qty(
                        abs(delta_qty), price, min_notional, step_size, risk.min_order_qty
                    )
                    if qty_scale < threshold:
                        log.info(
                            "Skipping order: qty %.8f below min threshold", abs(delta_qty)
                        )
                        log.info(
                            "METRICS %s",
                            json.dumps({"event": "skip", "reason": "below_min_qty"}),
                        )
                        continue
                    prev_pos_qty = pos_qty
                    try:
                        prev_rpnl = float(getattr(broker.state, "realized_pnl", 0.0))
                    except (TypeError, ValueError):
                        prev_rpnl = total_pnl
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
                    status = str(resp.get("status", ""))
                    filled_qty = float(resp.get("filled_qty", 0.0))
                    pending_qty = float(resp.get("pending_qty", 0.0))
                    if status == "rejected":
                        continue
                    log_order = False
                    order_qty = qty_scale
                    if status in {"open", "filled"}:
                        log_order = True
                    elif status == "canceled" and filled_qty > 0:
                        log_order = True
                        order_qty = filled_qty
                    if log_order:
                        log.info(
                            "METRICS %s",
                            json.dumps(
                                {
                                    "event": "order",
                                    "side": side,
                                    "price": price,
                                    "qty": order_qty,
                                    "fee": 0.0,
                                    "pnl": broker.state.realized_pnl,
                                }
                            ),
                        )
                        risk.account.update_open_order(symbol, side, pending_qty)
                        cur_qty = risk.account.current_exposure(symbol)[0]
                        locked = _recalc_locked_total()
                        log.info(
                            "METRICS %s",
                            json.dumps({"exposure": cur_qty, "locked": locked}),
                        )
                    risk.on_fill(
                        symbol,
                        side,
                        filled_qty,
                        venue=venue if not dry_run else "paper",
                    )
                    realized_raw = resp.get("realized_pnl", getattr(broker.state, "realized_pnl", prev_rpnl))
                    try:
                        realized_val = float(realized_raw)
                    except (TypeError, ValueError):
                        realized_val = prev_rpnl + 0.0
                    delta_rpnl = realized_val - prev_rpnl
                    try:
                        post_qty = float(risk.account.current_exposure(symbol)[0])
                    except Exception:
                        post_qty = pos_qty
                    if filled_qty > 0 and _position_closed(prev_pos_qty, post_qty):
                        _record_trade(delta_rpnl)
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
        if sig.side == "sell" and not risk.allow_short:
            cur_qty, _ = risk.account.current_exposure(symbol)
            if cur_qty <= 0:
                log.debug(
                    "Ignoring short signal while flat and shorting disabled for %s",
                    symbol,
                )
                continue
        signal_ts = getattr(sig, "signal_ts", time.time())
        pending = risk.account.open_orders.get(symbol, {}).get(sig.side, 0.0)
        allowed, reason, delta = risk.check_order(
            symbol,
            sig.side,
            closed.c,
            strength=sig.strength,
            volatility=bar.get("atr") or bar.get("volatility"),
            target_volatility=bar.get("target_volatility"),
            pending_qty=pending,
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
        threshold = _flat_threshold()
        if abs(delta) < threshold:
            log.info(
                "Skipping order: qty %.8f below min threshold", abs(delta)
            )
            log.info(
                "METRICS %s",
                json.dumps({"event": "skip", "reason": "below_min_qty"}),
            )
            continue
        side = "buy" if delta > 0 else "sell"
        price = (
            sig.limit_price
            if sig.limit_price is not None
            else limit_price_from_close(side, closed.c, tick_size)
        )
        qty = adjust_qty(abs(delta), price, min_notional, step_size, risk.min_order_qty)
        if qty < threshold:
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
        prev_pos_qty, _ = risk.account.current_exposure(symbol)
        try:
            prev_rpnl = float(getattr(broker.state, "realized_pnl", 0.0))
        except (TypeError, ValueError):
            prev_rpnl = total_pnl
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
        status = str(resp.get("status", ""))
        filled_qty = float(resp.get("filled_qty", 0.0))
        pending_qty = float(resp.get("pending_qty", 0.0))
        if status == "rejected":
            log.info("LIVE FILL %s", resp)
            continue
        log.info("LIVE FILL %s", resp)
        log_order = False
        order_qty = qty
        if status in {"open", "filled"}:
            log_order = True
        elif status == "canceled" and filled_qty > 0:
            log_order = True
            order_qty = filled_qty
        if log_order:
            log.info(
                "METRICS %s",
                json.dumps(
                    {
                        "event": "order",
                        "side": side,
                        "price": price,
                        "qty": order_qty,
                        "fee": 0.0,
                        "pnl": broker.state.realized_pnl,
                    }
                ),
            )
            risk.account.update_open_order(symbol, side, pending_qty)
            cur_qty = risk.account.current_exposure(symbol)[0]
            locked = _recalc_locked_total()
            log.info(
                "METRICS %s",
                json.dumps({"exposure": cur_qty, "locked": locked}),
        )
        risk.on_fill(
            symbol, side, filled_qty, venue=venue if not dry_run else "paper"
        )
        realized_raw = resp.get("realized_pnl", getattr(broker.state, "realized_pnl", prev_rpnl))
        try:
            realized_val = float(realized_raw)
        except (TypeError, ValueError):
            realized_val = prev_rpnl + 0.0
        delta_rpnl = realized_val - prev_rpnl
        try:
            post_qty = float(risk.account.current_exposure(symbol)[0])
        except Exception:
            post_qty = prev_pos_qty
        if filled_qty > 0 and _position_closed(prev_pos_qty, post_qty):
            _record_trade(delta_rpnl)
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

