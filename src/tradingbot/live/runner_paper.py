from __future__ import annotations
import asyncio
import errno
import logging
from datetime import datetime, timezone
import time
import contextlib
import json
import uvicorn
import ccxt

from sqlalchemy.exc import OperationalError

from .runner import Bar, BarAggregator
from ..adapters.binance import BinanceWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_futures import BinanceFuturesAdapter
from ..adapters.bybit_spot import BybitSpotAdapter
from ..adapters.bybit_futures import BybitFuturesAdapter
from ..adapters.okx_spot import OKXSpotAdapter as OKXSpotWSAdapter
from ..adapters.okx_spot import OKXSpotAdapter
from ..adapters.okx_futures import OKXFuturesAdapter
from ..execution.paper import PaperAdapter
from ..backtesting.engine import SlippageModel
from ..execution.router import ExecutionRouter
from ..utils.metrics import MARKET_LATENCY, AGG_COMPLETED, SKIPS, CANCELS
from ..utils.price import limit_price_from_close
from ..broker.broker import Broker
from ..config import settings
from ..risk.service import load_positions
from ..risk.portfolio_guard import GuardConfig, PortfolioGuard
from ..risk.service import RiskService
from ..risk.correlation_service import CorrelationService
from ..strategies import STRATEGIES
from monitoring import panel
from ..core.symbols import normalize
from ._symbol_rules import resolve_symbol_rules
from ..execution.order_sizer import adjust_qty
from ._metrics import infer_maker_flag

try:
    from ..storage.timescale import get_engine

    _CAN_PG = True
except Exception:  # pragma: no cover
    _CAN_PG = False

log = logging.getLogger(__name__)


_EXPOSURE_LOG_STATE: dict[tuple[str | None, str | None], tuple[float, float]] = {}


def _normalize_exposure_key(
    symbol: str | None, side: str | None
) -> tuple[str | None, str | None]:
    """Return a normalized key for exposure logging lookups."""

    sym_key: str | None
    if symbol is None or isinstance(symbol, str):
        sym_key = symbol
    else:
        sym_key = str(symbol)

    if isinstance(side, str):
        side_key: str | None = side.lower()
    elif side is None:
        side_key = None
    else:
        try:
            side_key = str(side).lower()
        except Exception:
            side_key = str(side)
    return sym_key, side_key


def _log_exposure_if_changed(
    symbol: str | None, side: str | None, exposure: float, locked: float
) -> bool:
    """Emit exposure metrics only when they change beyond tolerance."""

    key = _normalize_exposure_key(symbol, side)
    try:
        exposure_val = float(exposure)
    except (TypeError, ValueError):
        exposure_val = 0.0
    try:
        locked_val = float(locked)
    except (TypeError, ValueError):
        locked_val = 0.0
    prev = _EXPOSURE_LOG_STATE.get(key)
    if prev is not None:
        if (
            abs(prev[0] - exposure_val) <= 1e-9
            and abs(prev[1] - locked_val) <= 1e-9
        ):
            return False
    log.info(
        "METRICS %s",
        json.dumps({"exposure": exposure_val, "locked": locked_val}),
    )
    _EXPOSURE_LOG_STATE[key] = (exposure_val, locked_val)
    return True


def _reset_exposure_log(symbol: str | None, side: str | None) -> None:
    """Forget the last exposure/locked pair for ``symbol``/``side``."""

    key = _normalize_exposure_key(symbol, side)
    _EXPOSURE_LOG_STATE.pop(key, None)


def _clear_exposure_log_registry() -> None:
    """Clear all cached exposure metrics (primarily for tests)."""

    _EXPOSURE_LOG_STATE.clear()


WS_ADAPTERS = {
    ("binance", "spot"): BinanceSpotWSAdapter,
    ("binance", "futures"): BinanceWSAdapter,
    ("bybit", "spot"): BybitSpotAdapter,
    ("bybit", "futures"): BybitFuturesAdapter,
    ("okx", "spot"): OKXSpotWSAdapter,
    ("okx", "futures"): OKXFuturesAdapter,
}

REST_ADAPTERS = {
    ("binance", "spot"): BinanceSpotAdapter,
    ("binance", "futures"): BinanceFuturesAdapter,
    ("bybit", "spot"): BybitSpotAdapter,
    ("bybit", "futures"): BybitFuturesAdapter,
    ("okx", "spot"): OKXSpotAdapter,
    ("okx", "futures"): OKXFuturesAdapter,
}


_SPOT_MIN_QTY_FALLBACK = 1e-4
_SPOT_MIN_NOTIONAL_FALLBACK = 10.0


async def _start_metrics(port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    """Launch the monitoring panel in the background."""
    config = uvicorn.Config(panel.app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        task: asyncio.Task[None] = asyncio.create_task(server.serve())
    except SystemExit as exc:  # pragma: no cover - defensive
        raise OSError(errno.EADDRINUSE) from exc
    while not getattr(server, "started", False) and not task.done():
        await asyncio.sleep(0.1)
    if task.done():
        exc = task.exception()
        if exc is not None:
            if isinstance(exc, SystemExit):
                raise OSError(errno.EADDRINUSE) from exc
            raise exc
    return server, task


async def run_paper(
    symbol: str = "BTC/USDT",
    strategy_name: str = "breakout_atr",
    *,
    venue: str = "binance_spot",
    config_path: str | None = None,
    metrics_port: int = 8000,
    corr_threshold: float = 0.8,
    risk_pct: float = 0.0,
    params: dict | None = None,
    timeframe: str = "1m",
    initial_cash: float = 1000.0,
    risk_per_trade: float = 1.0,
    total_cap_pct: float | None = None,
    per_symbol_cap_pct: float | None = None,
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
    slippage_bps: float = 0.0,
    slip_bps_per_qty: float = 0.0,
    reprice_bps: float = 0.0,
    min_notional: float = 0.0,
    step_size: float = 0.0,
) -> None:
    """Run a simple live pipeline entirely in paper mode.

    Parameters
    ----------
    slip_bps_per_qty:
        Optional manual slippage in basis points applied per unit of traded
        quantity. Slippage is otherwise estimated automatically from order
        book depth or historical executions, so this parameter can usually be
        left at ``0.0``.
    """
    raw_symbol = symbol
    symbol = normalize(symbol)
    timeframe_seconds = ccxt.Exchange.parse_timeframe(timeframe)
    expiry = timeframe_seconds
    exchange, market = venue.split("_", 1)
    ws_cls = WS_ADAPTERS.get((exchange, market))
    if ws_cls is None:
        raise ValueError(f"unsupported venue: {venue}")
    log.info(
        "Connecting to %s %s WS in paper mode for %s",
        exchange.capitalize(),
        market,
        symbol,
    )
    # Allow slight delays without dropping the connection without mutating
    # global settings. Configure per-adapter ping timing instead.
    adapter = ws_cls()
    adapter.ping_interval = max(getattr(adapter, "ping_interval", 20.0), 30.0)
    timeout = getattr(adapter, "ping_timeout", 20.0) or 20.0
    adapter.ping_timeout = max(timeout, 30.0)

    rest = getattr(adapter, "rest", None)
    if rest is None:
        rest_cls = REST_ADAPTERS.get((exchange, market))
        if rest_cls is not None:
            rest = rest_cls()
            if hasattr(adapter, "rest"):
                adapter.rest = rest
    tick_size = 0.0
    min_qty_val = 0.0

    if rest is not None and hasattr(rest, "meta"):

        def _update_symbol_rules() -> None:
            nonlocal min_qty_val, min_notional, step_size, tick_size
            rules, _ = resolve_symbol_rules(rest.meta, raw_symbol, symbol)
            qty_step = float(getattr(rules, "qty_step", 0.0) or 0.0)
            if step_size <= 0 and qty_step > 0:
                step_size = qty_step
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

        def _current_missing() -> list[str]:
            missing: list[str] = []
            if step_size <= 0:
                missing.append("step_size")
            if min_qty_val <= 0:
                missing.append("min_qty")
            if min_notional <= 0 and market == "spot":
                missing.append("min_notional")
            return missing

        def _refresh_metadata() -> bool:
            load_markets = getattr(getattr(rest, "meta", None), "load_markets", None)
            if callable(load_markets):
                try:
                    load_markets()
                    return True
                except Exception as exc:  # pragma: no cover - defensive logging
                    log.warning("Failed to refresh markets for %s: %s", symbol, exc)
            return False

        for attempt in range(2):
            try:
                _update_symbol_rules()
            except Exception:
                if attempt == 0 and _refresh_metadata():
                    continue
                if step_size <= 0:
                    step_size = 1e-9
                tick_size = 0.0
                break

            missing_fields = _current_missing()
            if missing_fields:
                if attempt == 0 and _refresh_metadata():
                    continue
                hint_map = {
                    "step_size": "--step-size",
                    "min_qty": "--step-size",
                    "min_notional": "--min-notional",
                }
                missing_desc = []
                for field in missing_fields:
                    hint = hint_map.get(field)
                    if hint:
                        missing_desc.append(f"{field} (set via {hint})")
                    else:
                        missing_desc.append(field)
                details = ", ".join(missing_desc)
                raise RuntimeError(
                    "Missing symbol metadata for "
                    f"{symbol}: {details}. Reloaded markets but values remain unset; "
                    "provide explicit overrides via CLI."
                )
            break
    elif step_size <= 0:
        step_size = 1e-9
    import inspect
    slippage_model = SlippageModel(
        volume_impact=0.1,
        spread_mult=1.0,
        ofi_impact=0.0,
        source="bba",
        base_spread=0.0,
        pct=0.0001,
    )
    broker_kwargs = {
        "maker_fee_bps": maker_fee_bps,
        "taker_fee_bps": taker_fee_bps,
        "min_notional": min_notional,
        "step_size": step_size,
        "slip_bps_per_qty": slip_bps_per_qty,
        "slippage_model": slippage_model,
    }
    sig = inspect.signature(PaperAdapter)
    broker = PaperAdapter(
        **{k: v for k, v in broker_kwargs.items() if k in sig.parameters}
    )
    broker.state.cash = initial_cash
    if hasattr(broker.account, "update_cash"):
        broker.account.update_cash(initial_cash)
    try:
        broker.account.market_type = market
    except Exception:
        pass

    guard = PortfolioGuard(
        GuardConfig(
            total_cap_pct=total_cap_pct,
            per_symbol_cap_pct=per_symbol_cap_pct,
            venue="paper",
        )
    )
    guard.refresh_usd_caps(initial_cash)
    corr = CorrelationService()
    risk = RiskService(
        guard,
        corr_service=corr,
        account=broker.account,
        risk_pct=risk_pct,
        risk_per_trade=risk_per_trade,
        market_type=market,
    )
    min_qty_value = min_qty_val if min_qty_val > 0 else 0.0
    step_value = step_size if step_size > 0 else 0.0
    min_order_qty = max(min_qty_value, step_value)
    if min_order_qty <= 0:
        min_order_qty = _SPOT_MIN_QTY_FALLBACK if market == "spot" else 1e-9
    risk.min_order_qty = min_order_qty
    notional_value = float(min_notional) if min_notional and min_notional > 0 else 0.0
    if notional_value <= 0 and market == "spot":
        notional_value = _SPOT_MIN_NOTIONAL_FALLBACK
    risk.min_notional = float(notional_value)

    trades_closed = 0
    trades_won = 0
    pnl_won_total = 0.0
    pnl_lost_total = 0.0
    try:
        total_pnl = float(getattr(broker.state, "realized_pnl", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_pnl = 0.0

    logged_order_ids: set[str] = set()
    order_baselines: dict[str, tuple[float, float]] = {}

    def _capture_baseline(order_id: str | None, pnl_base: float, qty_base: float) -> None:
        if not order_id:
            return
        try:
            order_baselines[str(order_id)] = (float(pnl_base), float(qty_base))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    def _with_baseline(
        cb, pnl_base: float, qty_base: float
    ):
        if cb is None:
            return None

        def _inner(order, res):
            if order is not None:
                if not hasattr(order, "_pnl_baseline"):
                    try:
                        setattr(order, "_pnl_baseline", float(pnl_base))
                    except (TypeError, ValueError):
                        setattr(order, "_pnl_baseline", 0.0)
                if not hasattr(order, "_qty_before"):
                    try:
                        setattr(order, "_qty_before", float(qty_base))
                    except (TypeError, ValueError):
                        setattr(order, "_qty_before", 0.0)
            if isinstance(res, dict):
                key = res.get("order_id") or res.get("client_order_id")
                if key:
                    _capture_baseline(key, pnl_base, qty_base)
                    if order is not None:
                        setattr(order, "_baseline_key", str(key))
            return cb(order, res)

        return _inner

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
                guard.cfg.venue,
                sym,
                data.get("qty", 0.0),
                entry_price=data.get("avg_price"),
            )

    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")
    params = dict(params or {})
    leverage_value = params.pop("leverage", 1)
    log.info("METRICS %s", json.dumps({"leverage": leverage_value}))
    strat = (
        strat_cls(config_path=config_path, **params)
        if (config_path or params)
        else strat_cls()
    )
    strat.risk_service = risk

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
                for key, qty_raw in orders.items():
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

    def _prev_pending_qty(symbol: str | None, side: str | None) -> float:
        """Return the currently tracked pending quantity for ``symbol``/``side``."""

        if not symbol or not side:
            return 0.0

        account = getattr(risk, "account", None)
        if account is None:
            return 0.0

        open_orders = getattr(account, "open_orders", None)
        if not isinstance(open_orders, dict):
            return 0.0

        try:
            side_key = str(side).lower()
        except Exception:
            side_key = side

        try:
            existing = open_orders.get(symbol, {}).get(side_key, 0.0)
        except AttributeError:
            return 0.0

        try:
            return float(existing or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _make_order_ack_logger(
        symbol: str,
        side: str,
        price: float,
        qty_hint: float,
        prev_rpnl: float,
        prev_pos_qty: float,
    ):
        """Create a callback that emits metrics once an order is acknowledged."""

        qty_hint_val = float(qty_hint)

        def _on_ack(order, res):
            if not isinstance(res, dict):
                return
            status = str(res.get("status", "")).lower()
            if status in {"rejected", "error"}:
                return
            order_id_val = res.get("order_id") or res.get("client_order_id")
            if order_id_val is not None and str(order_id_val) in logged_order_ids:
                return

            px_val = res.get("price")
            if px_val is None and order is not None:
                px_val = getattr(order, "price", None)
            if px_val is None:
                px_val = price
            try:
                px_float = float(px_val)
            except (TypeError, ValueError):
                px_float = float(price)

            qty_val = res.get("pending_qty")
            if qty_val in (None, ""):
                qty_val = res.get("qty")
            if qty_val in (None, "") and order is not None:
                qty_val = getattr(order, "qty", None)
            if qty_val in (None, ""):
                qty_val = res.get("filled_qty")
            try:
                qty_float = float(qty_val)
            except (TypeError, ValueError):
                qty_float = qty_hint_val
            if qty_float <= 0:
                try:
                    qty_float = float(res.get("filled_qty") or qty_hint_val)
                except (TypeError, ValueError):
                    qty_float = qty_hint_val

            try:
                pnl_snapshot = float(getattr(broker.state, "realized_pnl", 0.0))
            except (TypeError, ValueError):
                pnl_snapshot = 0.0

            log.info(
                "METRICS %s",
                json.dumps(
                    {
                        "event": "order",
                        "side": side,
                        "price": px_float,
                        "qty": qty_float,
                        "fee": 0.0,
                        "pnl": pnl_snapshot,
                    }
                ),
            )
            if order_id_val is not None:
                logged_order_ids.add(str(order_id_val))
            _capture_baseline(order_id_val, prev_rpnl, prev_pos_qty)

            account = getattr(risk, "account", None)
            if account is None:
                return
            pending_qty = qty_float
            if pending_qty < 0.0 and abs(pending_qty) <= 1e-9:
                pending_qty = 0.0
            try:
                side_norm = str(side).lower()
            except Exception:
                side_norm = side
            prev_pending = _prev_pending_qty(symbol, side_norm)
            delta_pending = pending_qty - prev_pending
            if delta_pending < 0.0 and abs(delta_pending) <= 1e-9:
                delta_pending = 0.0
            update_open = getattr(account, "update_open_order", None)
            if (
                callable(update_open)
                and symbol
                and isinstance(symbol, str)
                and side_norm
            ):
                with contextlib.suppress(Exception):
                    update_open(symbol, side_norm, delta_pending)
            try:
                cur_qty = float(account.current_exposure(symbol)[0])
            except Exception:
                cur_qty = 0.0
            if step_size > 0 and abs(cur_qty) < step_size:
                cur_qty = 0.0
                try:
                    positions = getattr(account, "positions", None)
                    if isinstance(positions, dict):
                        positions[symbol] = 0.0
                except Exception:
                    pass
            locked = _recalc_locked_total()
            if not getattr(account, "open_orders", {}):
                locked = 0.0
            _log_exposure_if_changed(symbol, side, cur_qty, locked)

        return _on_ack

    def on_order_cancel(res: dict) -> None:
        """Handle broker order cancellation notifications."""
        if not isinstance(res, dict):
            return
        status = str(res.get("status", "")).lower()
        if status not in {"canceled", "cancelled", "expired"}:
            return
        if res.get("_cancel_handled"):
            return
        res["_cancel_handled"] = True
        symbol = res.get("symbol")
        side = res.get("side")
        side_norm = str(side).lower() if isinstance(side, str) else None
        lookup_side = side_norm or side
        pending_raw = res.get("pending_qty")
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
        venue = res.get("venue") if isinstance(res, dict) else None
        metric_pending = res.get("pending_qty", pending_qty)
        if metric_pending_override is not None:
            metric_pending = metric_pending_override
        try:
            metric_pending_val = float(metric_pending)
        except (TypeError, ValueError):
            metric_pending_val = 0.0
        side_for_risk = side_norm
        if not side_for_risk and isinstance(lookup_side, str):
            side_for_risk = lookup_side.lower()
        treat_as_fill = filled_qty > 0 or metric_pending_val <= 0
        if treat_as_fill:
            risk.complete_order(
                venue=venue,
                symbol=symbol,
                side=side_for_risk,
            )
            locked_total = _recalc_locked_total()
            if not getattr(risk.account, "open_orders", {}):
                locked_total = 0.0
            exposure_val = 0.0
            if symbol:
                current_exposure_fn = getattr(risk.account, "current_exposure", None)
                if callable(current_exposure_fn):
                    try:
                        exposure_val = float(current_exposure_fn(symbol)[0])
                    except Exception:
                        exposure_val = 0.0
            reset_side = side_for_risk if side_for_risk is not None else lookup_side
            _log_exposure_if_changed(symbol, reset_side, exposure_val, locked_total)
            _reset_exposure_log(symbol, reset_side)
            return
        CANCELS.inc()
        risk.complete_order(
            venue=venue,
            symbol=symbol,
            side=side_for_risk,
        )
        locked_total = _recalc_locked_total()
        order_id_val = None
        if isinstance(res, dict):
            order_id_val = res.get("order_id") or res.get("client_order_id")
        pending_metric = metric_pending_val
        if abs(pending_metric) <= 1e-9:
            pending_metric = 0.0
        log.info(
            "METRICS %s",
            json.dumps(
                {
                    "event": "cancel",
                    "reason": res.get("reason"),
                    "locked": locked_total,
                    "order_id": order_id_val,
                    "pending_qty": pending_metric,
                    "filled_qty": filled_qty,
                }
            ),
        )
        reset_side = side_for_risk if side_for_risk is not None else lookup_side
        _reset_exposure_log(symbol, reset_side)

    router = ExecutionRouter([
        broker
    ], prefer="maker")

    last_price = 0.0

    def _wrap_cb(orig_cb, *, call_cancel=False):
        def _cb(order, res):
            logged_exposure = False
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
                    on_order_cancel(res_dict)
            else:
                res = res_dict or res
            action = orig_cb(order, res) if orig_cb else None
            filled_qty = 0.0
            if isinstance(res, dict):
                try:
                    filled_qty = float(res.get("filled_qty", 0.0) or 0.0)
                except (TypeError, ValueError):
                    filled_qty = 0.0
            skip_completion = (
                call_cancel
                and status in {"canceled", "cancelled", "expired"}
                and filled_qty <= 0.0
            )
            if filled_qty > 0:
                symbol = None
                side = None
                if order is not None:
                    symbol = getattr(order, "symbol", None)
                    side = getattr(order, "side", None)
                if symbol is None:
                    symbol = res.get("symbol") if isinstance(res, dict) else None
                if side is None:
                    side = res.get("side") if isinstance(res, dict) else None
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
                base_price = None
                if order is not None:
                    base_price = getattr(order, "price", None)
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
                order_id = None
                if isinstance(res, dict):
                    order_id = res.get("order_id") or res.get("client_order_id")
                order_key = str(order_id) if order_id is not None else None
                if order_key and order_key not in logged_order_ids:
                    qty_for_event = None
                    if isinstance(res, dict):
                        qty_for_event = (
                            res.get("qty")
                            or res.get("filled_qty")
                            or res.get("orig_qty")
                        )
                    if qty_for_event is None and order is not None:
                        qty_for_event = getattr(order, "qty", None)
                    try:
                        qty_for_event = float(qty_for_event)
                    except (TypeError, ValueError):
                        qty_for_event = filled_qty
                    price_for_event = None
                    if isinstance(res, dict):
                        price_for_event = res.get("price") or res.get("avg_price")
                    if price_for_event is None and order is not None:
                        price_for_event = getattr(order, "price", None)
                    if price_for_event is None:
                        price_for_event = exec_price if exec_price is not None else base_price
                    try:
                        price_for_event = (
                            float(price_for_event)
                            if price_for_event is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        price_for_event = exec_price
                    log.info(
                        "METRICS %s",
                        json.dumps(
                            {
                                "event": "order",
                                "side": side,
                                "price": price_for_event,
                                "qty": qty_for_event,
                                "fee": 0.0,
                                "pnl": getattr(broker.state, "realized_pnl", 0.0),
                            }
                        ),
                    )
                    logged_order_ids.add(order_key)
                maker_flag = infer_maker_flag(
                    res if isinstance(res, dict) else None,
                    exec_price,
                    base_price,
                )
                pending_qty = None
                if isinstance(res, dict):
                    pending_raw = res.get("pending_qty")
                    if pending_raw is not None:
                        try:
                            pending_qty = float(pending_raw)
                        except (TypeError, ValueError):
                            pending_qty = None
                if pending_qty is not None and abs(pending_qty) <= 1e-9:
                    pending_qty = 0.0
                if isinstance(res, dict) and pending_qty is not None:
                    res["pending_qty"] = pending_qty
                metrics_payload = {
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
                    "order_id": order_id,
                    "pending_qty": pending_qty,
                    "filled_qty": filled_qty,
                }
                log.info("METRICS %s", json.dumps(metrics_payload))
                if symbol and side_norm:
                    pending_value = pending_qty
                    if isinstance(res, dict):
                        pending_raw = res.get("pending_qty")
                        if pending_raw is not None:
                            try:
                                pending_value = float(pending_raw)
                            except (TypeError, ValueError):
                                pending_value = None
                    if pending_value is not None and abs(pending_value) <= 1e-9:
                        pending_value = 0.0
                    if pending_value is not None and isinstance(res, dict):
                        res["pending_qty"] = pending_value
                    if pending_value is not None:
                        adapters_to_update: list[object] = []
                        for candidate in (adapter, rest):
                            if candidate is not None and candidate not in adapters_to_update:
                                adapters_to_update.append(candidate)
                        for candidate in adapters_to_update:
                            handler = getattr(candidate, "on_paper_fill", None)
                            if callable(handler):
                                try:
                                    handler(symbol, side_norm, pending_value)
                                except Exception:  # pragma: no cover - defensive
                                    pass
                    account_open_orders = getattr(risk.account, "open_orders", None)
                    update_open = getattr(risk.account, "update_open_order", None)
                    prev_pending = 0.0
                    if isinstance(account_open_orders, dict):
                        prev_pending = float(
                            account_open_orders.get(symbol, {}).get(side_norm, 0.0) or 0.0
                        )
                    if pending_value is not None and callable(update_open):
                        if isinstance(account_open_orders, dict):
                            delta_pending = pending_value - prev_pending
                            if abs(delta_pending) > 1e-12:
                                update_open(symbol, side_norm, delta_pending)
                        else:
                            update_open(symbol, side_norm, pending_value)
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
                    if not getattr(risk.account, "open_orders", {}):
                        locked = 0.0
                    _log_exposure_if_changed(symbol, side_norm, exposure_qty, locked)
                    logged_exposure = True
            if not skip_completion:
                pending_raw = res.get("pending_qty")
                pending_qty = None
                if pending_raw is not None:
                    try:
                        pending_qty = float(pending_raw)
                    except (TypeError, ValueError):
                        pending_qty = None
                if pending_qty is not None and abs(pending_qty) <= 1e-9:
                    pending_qty = 0.0
                    if isinstance(res, dict):
                        res["pending_qty"] = pending_qty
                if pending_qty is not None and pending_qty <= 0:
                    already_completed = False
                    if order is not None:
                        already_completed = getattr(order, "_risk_order_completed", False)
                    symbol_for_metrics = None
                    side_for_completion = None
                    venue_for_completion = None
                    if order is not None:
                        symbol_for_metrics = getattr(order, "symbol", None)
                        side_for_completion = getattr(order, "side", None)
                        venue_for_completion = getattr(order, "venue", None)
                    if isinstance(res, dict):
                        if symbol_for_metrics is None:
                            symbol_for_metrics = res.get("symbol")
                        if side_for_completion is None:
                            side_for_completion = res.get("side")
                        if venue_for_completion is None:
                            venue_for_completion = res.get("venue")
                    if isinstance(side_for_completion, str):
                        side_for_completion = side_for_completion.lower()
                    else:
                        side_for_completion = None
                    if not already_completed:
                        risk.complete_order(
                            venue=venue_for_completion,
                            symbol=symbol_for_metrics,
                            side=side_for_completion,
                        )
                        if order is not None:
                            setattr(order, "_risk_order_completed", True)
                    locked_after_completion = _recalc_locked_total()
                    if not getattr(risk.account, "open_orders", {}):
                        locked_after_completion = 0.0
                    exposure_after = 0.0
                    if symbol_for_metrics:
                        try:
                            exposure_after = float(
                                risk.account.current_exposure(symbol_for_metrics)[0]
                            )
                        except Exception:
                            exposure_after = 0.0
                    if not logged_exposure:
                        _log_exposure_if_changed(
                            symbol_for_metrics,
                            side_for_completion,
                            exposure_after,
                            locked_after_completion,
                        )
                    _reset_exposure_log(symbol_for_metrics, side_for_completion)
                    baseline_qty = None
                    baseline_pnl = None
                    baseline_key = None
                    if order is not None:
                        baseline_qty = getattr(order, "_qty_before", None)
                        baseline_pnl = getattr(order, "_pnl_baseline", None)
                        baseline_key = getattr(order, "_baseline_key", None)
                    lookup_key = None
                    if isinstance(res, dict):
                        lookup_key = res.get("order_id") or res.get("client_order_id")
                    if lookup_key is None:
                        lookup_key = baseline_key
                    if lookup_key is not None and (
                        baseline_qty is None or baseline_pnl is None
                    ):
                        stored = order_baselines.get(str(lookup_key))
                        if stored:
                            if baseline_pnl is None:
                                baseline_pnl = stored[0]
                                if order is not None:
                                    setattr(order, "_pnl_baseline", stored[0])
                            if baseline_qty is None:
                                baseline_qty = stored[1]
                                if order is not None:
                                    setattr(order, "_qty_before", stored[1])
                            if order is not None and getattr(order, "_baseline_key", None) is None:
                                setattr(order, "_baseline_key", str(lookup_key))
                    try:
                        after_qty_val = float(exposure_after)
                    except (TypeError, ValueError):
                        after_qty_val = None
                    before_qty_val = None
                    if baseline_qty is not None:
                        try:
                            before_qty_val = float(baseline_qty)
                        except (TypeError, ValueError):
                            before_qty_val = None
                    if (
                        before_qty_val is not None
                        and after_qty_val is not None
                        and baseline_pnl is not None
                        and _position_closed(before_qty_val, after_qty_val)
                    ):
                        already_recorded = False
                        if order is not None:
                            already_recorded = getattr(order, "_trade_recorded", False)
                        if not already_recorded:
                            pnl_source = None
                            if isinstance(res, dict):
                                pnl_source = res.get("realized_pnl")
                            if pnl_source is None:
                                pnl_source = getattr(broker.state, "realized_pnl", None)
                            try:
                                current_pnl = float(pnl_source)
                                delta_pnl = current_pnl - float(baseline_pnl)
                            except (TypeError, ValueError):
                                delta_pnl = None
                            if delta_pnl is not None:
                                _record_trade(delta_pnl)
                                if order is not None:
                                    setattr(order, "_trade_recorded", True)
                                key_to_clear = lookup_key
                                if key_to_clear is not None:
                                    order_baselines.pop(str(key_to_clear), None)
                                log.info("METRICS %s", json.dumps({"pnl": total_pnl}))
            if action in {"re_quote", "requote", "re-quote"}:
                return None
            return action
        return _cb

    on_pf = _wrap_cb(strat.on_partial_fill)
    on_oe = _wrap_cb(strat.on_order_expiry, call_cancel=True)
    router.on_partial_fill = on_pf
    router.on_order_expiry = on_oe
    exec_broker = Broker(router)
    tif = f"GTD:{expiry}|PO"

    metrics_task: asyncio.Task[None] | None = None
    port = metrics_port
    while True:
        try:
            server, metrics_task = await _start_metrics(port)
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

    warmup_total = getattr(strat, "warmup_bars", 140)
    try:
        agg = BarAggregator(timeframe=timeframe)
    except TypeError:
        agg = BarAggregator()
    if not hasattr(agg, "completed"):
        agg.completed = []

    if rest is not None and warmup_total > 0:
        try:
            log.info("Loading %d historical bars for warm-up", warmup_total)
            fetch_symbol = symbol
            if exchange == "okx" and hasattr(adapter, "normalize_symbol"):
                fetch_symbol = adapter.normalize_symbol(symbol).replace("-", "/")
            if hasattr(rest, "fetch_ohlcv"):
                bars = await rest.fetch_ohlcv(
                    fetch_symbol, timeframe=timeframe, limit=warmup_total
                )
            else:
                client = rest.rest if hasattr(rest, "rest") else rest
                bars = await client.fetch_ohlcv(
                    fetch_symbol, timeframe=timeframe, limit=warmup_total
                )
            for ts_ms, o, h, l, c, v in bars:
                ts_bar = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
                agg.completed.append(Bar(ts_open=ts_bar, o=o, h=h, l=l, c=c, v=v))
            log.info("Pre-loaded %d/%d bars", len(agg.completed), warmup_total)
        except Exception as e:  # pragma: no cover - best effort
            log.warning("Failed to pre-load historical bars: %s", e)

    purge_interval = settings.risk_purge_minutes * 60.0
    last_purge = time.time()

    last_progress = len(agg.completed)
    last_log = 0
    prev_bars = len(agg.completed)

    try:
        async for t in adapter.stream_trades(symbol):
            ts = t.get("ts") or datetime.now(timezone.utc)
            px = float(t.get("price"))
            qty = float(t.get("qty", 0.0))
            last_price = px
            log.debug(
                "METRICS %s",
                json.dumps(
                    {
                        "event": "trade",
                        "price": px,
                        "qty": qty,
                        "pnl": broker.state.realized_pnl,
                    }
                ),
            )
            events = broker.update_last_price(symbol, px)
            for ev in events or []:
                await router.handle_paper_event(ev)
            risk.mark_price(symbol, px)
            if time.time() - last_purge >= purge_interval:
                risk.purge([symbol])
                last_purge = time.time()
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
                    try:
                        prev_rpnl = float(getattr(broker.state, "realized_pnl", 0.0))
                    except (TypeError, ValueError):
                        prev_rpnl = total_pnl
                    last_close = agg.completed[-1].c if agg.completed else px
                    price = limit_price_from_close(close_side, last_close, tick_size)
                    qty_close = adjust_qty(
                        abs(pos_qty), price, min_notional, step_size, risk.min_order_qty
                    )
                    qty_close = min(qty_close, abs(pos_qty))
                    if qty_close < threshold:
                        log.info(
                            "Skipping order: qty %.8f below min threshold", abs(pos_qty)
                        )
                        SKIPS.inc()
                        log.info(
                            "METRICS %s",
                            json.dumps({"event": "skip", "reason": "below_min_qty"}),
                        )
                        continue
                    pf_wrapped = _with_baseline(on_pf, prev_rpnl, prev_pos_qty)
                    oe_wrapped = _with_baseline(on_oe, prev_rpnl, prev_pos_qty)
                    ack_cb = _make_order_ack_logger(
                        symbol,
                        close_side,
                        price,
                        qty_close,
                        prev_rpnl,
                        prev_pos_qty,
                    )
                    resp = await exec_broker.place_limit(
                        symbol,
                        close_side,
                        price,
                        qty_close,
                        tif=tif,
                        on_partial_fill=pf_wrapped,
                        on_order_expiry=oe_wrapped,
                        on_order_ack=ack_cb,
                        slip_bps=slippage_bps,
                    )
                    status = str(resp.get("status", ""))
                    filled_qty = float(resp.get("filled_qty", 0.0))
                    pending_qty = float(resp.get("pending_qty", 0.0))
                    if abs(pending_qty) <= 1e-9:
                        pending_qty = 0.0
                    order_id_val = resp.get("order_id") or resp.get("client_order_id")
                    exec_price = float(resp.get("price", price))
                    if status == "rejected":
                        if resp.get("reason") == "insufficient_cash":
                            SKIPS.inc()
                            log.info(
                                "METRICS %s",
                                json.dumps({"event": "skip", "reason": "insufficient_cash"}),
                            )
                        continue
                    log_order = False
                    order_qty = qty_close
                    if status in {"open", "filled"}:
                        log_order = True
                    elif status == "canceled" and filled_qty > 0:
                        log_order = True
                        order_qty = filled_qty
                    if log_order:
                        should_log_order = True
                        oid = resp.get("order_id")
                        if oid is not None and str(oid) in logged_order_ids:
                            should_log_order = False
                        if should_log_order:
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
                            if oid is not None:
                                logged_order_ids.add(str(oid))
                        _capture_baseline(
                            resp.get("order_id") or resp.get("client_order_id"),
                            prev_rpnl,
                            prev_pos_qty,
                        )
                        prev_pending = _prev_pending_qty(symbol, close_side)
                        delta_pending = pending_qty - prev_pending
                        risk.account.update_open_order(symbol, close_side, delta_pending)
                        cur_qty = risk.account.current_exposure(symbol)[0]
                        if step_size > 0 and abs(cur_qty) < step_size:
                            cur_qty = 0.0
                            risk.account.positions[symbol] = 0.0
                        locked = _recalc_locked_total()
                        if not getattr(risk.account, "open_orders", {}):
                            locked = 0.0
                        log.info(
                            "METRICS %s",
                            json.dumps({"exposure": cur_qty, "locked": locked}),
                        )
                    realized_raw = resp.get(
                        "realized_pnl", getattr(broker.state, "realized_pnl", prev_rpnl)
                    )
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
                    delta_rpnl = realized_val - prev_rpnl
                    if filled_qty > 0:
                        slippage = (
                            ((exec_price - price) / price) * 10000 if price else 0.0
                        )
                        maker = exec_price == price
                        fee_bps = (
                            exec_broker.maker_fee_bps if maker else broker.taker_fee_bps
                        )
                        fee = filled_qty * exec_price * (fee_bps / 10000.0)
                        log.info(
                            "METRICS %s",
                            json.dumps(
                                {
                                    "event": "fill",
                                    "side": close_side,
                                    "price": exec_price,
                                    "qty": filled_qty,
                                    "fee": fee,
                                    "pnl": delta_rpnl,
                                    "slippage_bps": slippage,
                                    "maker": maker,
                                    "order_id": order_id_val,
                                    "pending_qty": pending_qty,
                                    "filled_qty": filled_qty,
                                }
                            ),
                        )
                    delta_rpnl = realized_val - prev_rpnl
                    halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                    if halted:
                        log.error("[HALT] motivo=%s", reason)
                        break
                    continue
                if decision in {"scale_in", "scale_out"}:
                    target = risk.calc_position_size(
                        trade.get("strength", 1.0), px, clamp=False
                    )
                    delta_qty = target - abs(pos_qty)
                    if abs(delta_qty) > threshold:
                        side = (
                            trade["side"]
                            if delta_qty > 0
                            else ("sell" if trade["side"] == "buy" else "buy")
                        )
                        prev_pos_qty = pos_qty
                        try:
                            prev_rpnl = float(getattr(broker.state, "realized_pnl", 0.0))
                        except (TypeError, ValueError):
                            prev_rpnl = total_pnl
                        last_close = agg.completed[-1].c if agg.completed else px
                        price = limit_price_from_close(side, last_close, tick_size)
                        qty_scale = abs(delta_qty)
                        qty_scale = min(qty_scale, abs(pos_qty))
                        if qty_scale < threshold:
                            log.info(
                                "Skipping order: qty %.8f below min threshold",
                                abs(delta_qty),
                            )
                            SKIPS.inc()
                            log.info(
                                "METRICS %s",
                                json.dumps({"event": "skip", "reason": "below_min_qty"}),
                            )
                            continue
                        qty_scale = adjust_qty(
                            qty_scale, price, min_notional, step_size, risk.min_order_qty
                        )
                        if qty_scale < threshold:
                            log.info(
                                "Skipping order: qty %.8f below min threshold",
                                abs(delta_qty),
                            )
                            SKIPS.inc()
                            log.info(
                                "METRICS %s",
                                json.dumps({"event": "skip", "reason": "below_min_qty"}),
                            )
                            continue
                        pf_wrapped = _with_baseline(on_pf, prev_rpnl, prev_pos_qty)
                        oe_wrapped = _with_baseline(on_oe, prev_rpnl, prev_pos_qty)
                        ack_cb = _make_order_ack_logger(
                            symbol,
                            side,
                            price,
                            qty_scale,
                            prev_rpnl,
                            prev_pos_qty,
                        )
                        resp = await exec_broker.place_limit(
                            symbol,
                            side,
                            price,
                            qty_scale,
                            tif=tif,
                            on_partial_fill=pf_wrapped,
                            on_order_expiry=oe_wrapped,
                            on_order_ack=ack_cb,
                            slip_bps=slippage_bps,
                        )
                        status = str(resp.get("status", ""))
                        filled_qty = float(resp.get("filled_qty", 0.0))
                        pending_qty = float(resp.get("pending_qty", 0.0))
                        if abs(pending_qty) <= 1e-9:
                            pending_qty = 0.0
                        order_id_val = resp.get("order_id") or resp.get("client_order_id")
                        exec_price = float(resp.get("price", price))
                        if status == "rejected":
                            if resp.get("reason") == "insufficient_cash":
                                SKIPS.inc()
                                log.info(
                                    "METRICS %s",
                                    json.dumps({"event": "skip", "reason": "insufficient_cash"}),
                                )
                            continue
                        log_order = False
                        order_qty = qty_scale
                        if status in {"open", "filled"}:
                            log_order = True
                        elif status == "canceled" and filled_qty > 0:
                            log_order = True
                            order_qty = filled_qty
                        if log_order:
                            should_log_order = True
                            oid = resp.get("order_id")
                            if oid is not None and str(oid) in logged_order_ids:
                                should_log_order = False
                            if should_log_order:
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
                                if oid is not None:
                                    logged_order_ids.add(str(oid))
                            _capture_baseline(
                                resp.get("order_id") or resp.get("client_order_id"),
                                prev_rpnl,
                                prev_pos_qty,
                            )
                            prev_pending = _prev_pending_qty(symbol, side)
                            delta_pending = pending_qty - prev_pending
                            risk.account.update_open_order(symbol, side, delta_pending)
                            cur_qty = risk.account.current_exposure(symbol)[0]
                            if step_size > 0 and abs(cur_qty) < step_size:
                                cur_qty = 0.0
                                risk.account.positions[symbol] = 0.0
                            locked = _recalc_locked_total()
                            if not getattr(risk.account, "open_orders", {}):
                                locked = 0.0
                            log.info(
                                "METRICS %s",
                                json.dumps({"exposure": cur_qty, "locked": locked}),
                            )
                        realized_raw = resp.get(
                            "realized_pnl", getattr(broker.state, "realized_pnl", prev_rpnl)
                        )
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
                        if filled_qty > 0:
                            slippage = (
                                ((exec_price - price) / price) * 10000 if price else 0.0
                            )
                            maker = exec_price == price
                            fee_bps = (
                                exec_broker.maker_fee_bps
                                if maker
                                else broker.taker_fee_bps
                            )
                            fee = filled_qty * exec_price * (fee_bps / 10000.0)
                            log.info(
                                "METRICS %s",
                                json.dumps(
                                    {
                                        "event": "fill",
                                        "side": side,
                                        "price": exec_price,
                                        "qty": filled_qty,
                                        "fee": fee,
                                        "pnl": delta_rpnl,
                                        "slippage_bps": slippage,
                                        "maker": maker,
                                        "order_id": order_id_val,
                                        "pending_qty": pending_qty,
                                        "filled_qty": filled_qty,
                                    }
                                ),
                            )
                        halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
                        if halted:
                            log.error("[HALT] motivo=%s", reason)
                            break
                    continue
            closed = agg.on_trade(ts, px, qty)
            latency = (datetime.now(timezone.utc) - ts).total_seconds()
            bars = len(agg.completed)
            MARKET_LATENCY.observe(latency)
            AGG_COMPLETED.set(bars)
            log.debug("bars accumulated=%d", bars)
            if bars != prev_bars and bars < warmup_total and bars % 10 == 0:
                log.info("Warm-up progress %d/%d", bars, warmup_total)
                prev_bars = bars
            if closed is None:
                continue
            correlations = await asyncio.to_thread(corr.get_correlations)
            risk.update_correlation(correlations, corr_threshold)
            df = await asyncio.to_thread(agg.last_n_bars_df, 200)
            progress = len(df)
            now = time.monotonic()
            if progress < warmup_total:
                if progress != last_progress or now - last_log >= 5:
                    log.info("Warm-up progress %d/%d", progress, warmup_total)
                    last_progress, last_log = progress, now
                continue
            bar = {"window": df, "symbol": symbol}
            signal = strat.on_bar(bar)
            if signal is None:
                continue
            if signal.side == "sell" and not risk.allow_short:
                cur_qty, _ = risk.account.current_exposure(symbol)
                if cur_qty <= 0:
                    log.debug(
                        "Ignoring short signal while flat and shorting disabled"
                    )
                    continue
            signal_ts = getattr(signal, "signal_ts", time.time())
            pending = risk.account.open_orders.get(symbol, {}).get(
                signal.side, 0.0
            )
            allowed, reason, delta = risk.check_order(
                symbol,
                signal.side,
                closed.c,
                strength=signal.strength,
                volatility=bar.get("atr") or bar.get("volatility"),
                target_volatility=bar.get("target_volatility"),
                pending_qty=pending,
            )
            if not allowed:
                if reason == "below_min_qty":
                    log.info(
                        "Skipping order: qty %.8f below min threshold", abs(delta)
                    )
                    SKIPS.inc()
                    log.info(
                        "METRICS %s",
                        json.dumps({"event": "skip", "reason": "below_min_qty"}),
                    )
                else:
                    log.warning("orden bloqueada: %s", reason)
                    SKIPS.inc()
                    log.info(
                        "METRICS %s",
                        json.dumps({"event": "risk_check_reject", "reason": reason}),
                    )
                    log.info(
                        "METRICS %s",
                        json.dumps({"event": "skip", "reason": reason}),
                    )
                continue
            side = "buy" if delta > 0 else "sell"
            price = getattr(signal, "limit_price", None)
            price = (
                price
                if price is not None
                else limit_price_from_close(side, closed.c, tick_size)
            )
            threshold = _flat_threshold()
            if abs(delta) < threshold:
                log.info(
                    "Skipping order: qty %.8f below min threshold", abs(delta)
                )
                SKIPS.inc()
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "skip", "reason": "below_min_qty"}),
                )
                continue
            qty = adjust_qty(
                abs(delta), price, min_notional, step_size, risk.min_order_qty
            )
            if qty < threshold:
                log.info(
                    "Skipping order: qty %.8f below min threshold", abs(delta)
                )
                SKIPS.inc()
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "skip", "reason": "below_min_qty"}),
                )
                continue
            notional = qty * price
            if notional < min_notional:
                reason = "below_min_notional"
                log.info(
                    "Skipping order: qty %.8f notional %.8f below min threshold", qty, notional
                )
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "skip", "reason": reason}),
                )
                continue
            if not risk.register_order(symbol, notional):
                reason = getattr(risk, "last_kill_reason", "register_reject")
                log.warning("registro de orden bloqueado: %s", reason)
                SKIPS.inc()
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "register_order_reject", "reason": reason}),
                )
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "skip", "reason": reason}),
                )
                continue
            prev_pos_qty, _ = risk.account.current_exposure(symbol)
            try:
                prev_rpnl = float(getattr(broker.state, "realized_pnl", 0.0))
            except (TypeError, ValueError):
                prev_rpnl = total_pnl
            pf_wrapped = _with_baseline(on_pf, prev_rpnl, prev_pos_qty)
            oe_wrapped = _with_baseline(on_oe, prev_rpnl, prev_pos_qty)
            ack_cb = _make_order_ack_logger(
                symbol,
                side,
                price,
                qty,
                prev_rpnl,
                prev_pos_qty,
            )
            resp = await exec_broker.place_limit(
                symbol,
                side,
                price,
                qty,
                tif=tif,
                on_partial_fill=pf_wrapped,
                on_order_expiry=oe_wrapped,
                on_order_ack=ack_cb,
                signal_ts=signal_ts,
                slip_bps=slippage_bps,
            )
            status = str(resp.get("status", ""))
            filled_qty = float(resp.get("filled_qty", 0.0))
            pending_qty = float(resp.get("pending_qty", 0.0))
            if abs(pending_qty) <= 1e-9:
                pending_qty = 0.0
            order_id_val = resp.get("order_id") or resp.get("client_order_id")
            exec_price = float(resp.get("price", price))
            if status == "rejected":
                if resp.get("reason") == "insufficient_cash":
                    SKIPS.inc()
                    log.info(
                        "METRICS %s",
                        json.dumps({"event": "skip", "reason": "insufficient_cash"}),
                    )
                continue
            log_order = False
            order_qty = qty
            if status in {"open", "filled", "new"}:
                log_order = True
            elif status == "canceled" and filled_qty > 0:
                log_order = True
                order_qty = filled_qty
            if log_order:
                should_log_order = True
                oid = resp.get("order_id")
                if oid is not None and str(oid) in logged_order_ids:
                    should_log_order = False
                if should_log_order:
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
                    if oid is not None:
                        logged_order_ids.add(str(oid))
                _capture_baseline(
                    resp.get("order_id") or resp.get("client_order_id"),
                    prev_rpnl,
                    prev_pos_qty,
                )
                prev_pending = _prev_pending_qty(symbol, side)
                delta_pending = pending_qty - prev_pending
                risk.account.update_open_order(symbol, side, delta_pending)
                cur_qty = risk.account.current_exposure(symbol)[0]
                if step_size > 0 and abs(cur_qty) < step_size:
                    cur_qty = 0.0
                    risk.account.positions[symbol] = 0.0
                locked = _recalc_locked_total()
                if not getattr(risk.account, "open_orders", {}):
                    locked = 0.0
                log.info(
                    "METRICS %s",
                    json.dumps({"exposure": cur_qty, "locked": locked}),
                )
            realized_raw = resp.get(
                "realized_pnl", getattr(broker.state, "realized_pnl", prev_rpnl)
            )
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
            if filled_qty > 0:
                slippage = ((exec_price - price) / price) * 10000 if price else 0.0
                maker = exec_price == price
                fee_bps = exec_broker.maker_fee_bps if maker else broker.taker_fee_bps
                fee = filled_qty * exec_price * (fee_bps / 10000.0)
                log.info(
                    "METRICS %s",
                    json.dumps(
                        {
                            "event": "fill",
                            "side": side,
                            "price": exec_price,
                            "qty": filled_qty,
                            "fee": fee,
                            "pnl": delta_rpnl,
                            "slippage_bps": slippage,
                            "maker": maker,
                            "order_id": order_id_val,
                            "pending_qty": pending_qty,
                            "filled_qty": filled_qty,
                        }
                    ),
                )
            halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
            if halted:
                log.error("[HALT] motivo=%s", reason)
                break
            continue
    finally:
        server.should_exit = True
        if metrics_task is not None:
            metrics_task.cancel()
            with contextlib.suppress(Exception):
                await metrics_task
