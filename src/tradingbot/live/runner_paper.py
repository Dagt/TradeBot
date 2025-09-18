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
from ..utils.metrics import MARKET_LATENCY, AGG_COMPLETED, SKIPS
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
from ..execution.order_sizer import adjust_qty
from .paper_orders import PaperOrderManager

try:
    from ..storage.timescale import get_engine

    _CAN_PG = True
except Exception:  # pragma: no cover
    _CAN_PG = False

log = logging.getLogger(__name__)


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
    try:
        if rest is not None and hasattr(rest, "meta"):
            fetch_symbol = None
            symbols = getattr(rest.meta.client, "symbols", [])
            if symbols:
                fetch_symbol = next(
                    (s for s in symbols if normalize(s) == symbol), None
                )
            if fetch_symbol is None:
                fetch_symbol = raw_symbol.replace("-", "/")
            rules = rest.meta.rules_for(fetch_symbol)
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
        elif step_size <= 0:
            step_size = 1e-9
    except Exception:
        if step_size <= 0:
            step_size = 1e-9
        tick_size = 0.0
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
    min_order_qty = min_qty_val if min_qty_val > 0 else 0.0
    if min_order_qty <= 0 and step_size > 0:
        min_order_qty = step_size
    risk.min_order_qty = min_order_qty if min_order_qty > 0 else 1e-9
    risk.min_notional = float(min_notional if min_notional > 0 else 0.0)

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
        step = step_size if step_size > 0 else 0.0
        if step > base:
            base = step
        return base if base > 0 else 1e-9

    def _position_closed(before: float, after: float) -> bool:
        threshold = _flat_threshold()
        if abs(before) <= threshold:
            return False
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
        payload = {
            "event": "trade",
            "pnl": float(delta_pnl),
            "pnl_trade": float(delta_pnl),
            "trade_pnl": float(delta_pnl),
            "trades_closed": trades_closed,
            "trades_won": trades_won,
            "pnl_won": pnl_won_total,
            "pnl_lost": pnl_lost_total,
            "expectancy": expectancy,
            "payoff_ratio": payoff_ratio,
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

    def _emit_metrics(payload: dict) -> None:
        log.info("METRICS %s", json.dumps(payload))

    def _get_exposure(symbol: str | None) -> float:
        if not symbol:
            return 0.0
        try:
            return float(risk.account.current_exposure(symbol)[0])
        except Exception:
            return 0.0

    order_manager = PaperOrderManager(
        emit=_emit_metrics,
        exposure_fn=_get_exposure,
    )

    def _sync_locked_total() -> None:
        account = getattr(risk, "account", None)
        if account is not None:
            try:
                setattr(account, "locked_total", order_manager.locked_total)
            except Exception:
                pass
        try:
            setattr(risk, "locked_total", order_manager.locked_total)
        except Exception:
            pass

    def _update_account_pending(symbol: str, side: str) -> None:
        side_norm = side.lower()
        account = getattr(risk, "account", None)
        if account is None:
            _sync_locked_total()
            return
        open_orders = getattr(account, "open_orders", {})
        prev_total = 0.0
        if isinstance(open_orders, dict):
            try:
                prev_total = float(open_orders.get(symbol, {}).get(side_norm, 0.0) or 0.0)
            except (TypeError, ValueError):
                prev_total = 0.0
        new_total = order_manager.total_remaining(symbol, side_norm)
        delta = new_total - prev_total
        update_open = getattr(account, "update_open_order", None)
        if callable(update_open) and abs(delta) > 1e-12:
            update_open(symbol, side_norm, delta)
        _sync_locked_total()

    router = ExecutionRouter([
        broker
    ], prefer="maker")

    last_price = 0.0

    def _wrap_cb(orig_cb, *, call_cancel: bool = False):
        def _cb(order, res):
            res_dict = res if isinstance(res, dict) else {}
            status = str(res_dict.get("status", "")).lower()
            action = orig_cb(order, res_dict) if orig_cb else None

            order_id_val = res_dict.get("order_id") or res_dict.get("client_order_id")
            if order_id_val is None and order is not None:
                order_id_val = getattr(order, "order_id", None)
            order_id = str(order_id_val) if order_id_val is not None else None

            symbol = res_dict.get("symbol")
            if symbol is None and order is not None:
                symbol = getattr(order, "symbol", None)

            side = res_dict.get("side")
            if side is None and order is not None:
                side = getattr(order, "side", None)
            side_norm = str(side).lower() if isinstance(side, str) else None

            filled_qty = 0.0
            try:
                filled_qty = float(res_dict.get("filled_qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                filled_qty = 0.0

            pending_qty_val = res_dict.get("pending_qty")
            try:
                pending_qty = float(pending_qty_val)
            except (TypeError, ValueError, TypeError):
                pending_qty = None
            if pending_qty is not None and abs(pending_qty) <= 1e-9:
                pending_qty = 0.0

            price_val = res_dict.get("price") or res_dict.get("avg_price")
            if price_val is None and order is not None:
                price_val = getattr(order, "price", None)
            try:
                exec_price = float(price_val) if price_val is not None else None
            except (TypeError, ValueError):
                exec_price = None

            base_price = None
            if order is not None:
                base_price = getattr(order, "price", None)

            slippage_bps = res_dict.get("slippage_bps")
            if slippage_bps is None and exec_price is not None and base_price:
                try:
                    base_price_f = float(base_price)
                    if base_price_f:
                        slippage_bps = ((exec_price - base_price_f) / base_price_f) * 10000.0
                except (TypeError, ValueError, ZeroDivisionError):
                    slippage_bps = None

            fee = res_dict.get("fee")
            fee_bps = res_dict.get("fee_bps")
            maker = None
            fee_type = res_dict.get("fee_type")
            if fee_type is not None:
                maker = str(fee_type).lower() == "maker"
            if fee is None and exec_price is not None and filled_qty > 0:
                if fee_bps is None:
                    maker_fee = getattr(exec_broker, "maker_fee_bps", getattr(broker, "maker_fee_bps", 0.0))
                    taker_fee = getattr(exec_broker, "taker_fee_bps", getattr(broker, "taker_fee_bps", 0.0))
                    if maker is None and base_price is not None:
                        try:
                            maker = abs(exec_price - float(base_price)) <= 1e-9
                        except (TypeError, ValueError):
                            maker = None
                    fee_bps = maker_fee if maker else taker_fee
                try:
                    fee = filled_qty * exec_price * (float(fee_bps) / 10000.0)
                except (TypeError, ValueError):
                    fee = None

            if symbol and order_id and order is not None:
                try:
                    orig_qty_val = getattr(order, "qty", None)
                    if orig_qty_val is None:
                        orig_qty_val = res_dict.get("orig_qty")
                    if orig_qty_val is None and pending_qty is not None:
                        orig_qty_val = pending_qty + filled_qty
                    if orig_qty_val is None:
                        orig_qty_val = filled_qty
                    remain_val = pending_qty
                    if remain_val is None and orig_qty_val is not None:
                        remain_val = max(float(orig_qty_val) - filled_qty, 0.0)
                    order_price = base_price if base_price is not None else exec_price
                    price_hint = res_dict.get("price")
                    if order_price is None and isinstance(price_hint, (int, float)):
                        order_price = float(price_hint)
                    side_for_event = side if side is not None else getattr(order, "side", None)
                    if order_price is not None and orig_qty_val is not None and side_for_event is not None:
                        order_manager.ensure_order_event(
                            symbol=symbol,
                            order_id=order_id,
                            side=str(side_for_event),
                            price=float(order_price),
                            orig_qty=float(orig_qty_val),
                            remaining_qty=float(remain_val or 0.0),
                        )
                except Exception:
                    pass

            if filled_qty > 0 and symbol and side_norm:
                if pending_qty is not None:
                    adapters_to_update: list[object] = []
                    for candidate in (adapter, rest):
                        if candidate is not None and candidate not in adapters_to_update:
                            adapters_to_update.append(candidate)
                    for candidate in adapters_to_update:
                        handler = getattr(candidate, "on_paper_fill", None)
                        if callable(handler):
                            try:
                                handler(symbol, side_norm, pending_qty)
                            except Exception:
                                pass
                update_position = getattr(risk.account, "update_position", None)
                if callable(update_position):
                    direction = 1.0 if side_norm == "buy" else -1.0
                    update_position(symbol, direction * filled_qty, price=exec_price)
                entry = order_manager.on_fill(
                    symbol=symbol,
                    order_id=order_id,
                    side=side if side is not None else side_norm,
                    fill_qty=filled_qty,
                    price=exec_price,
                    fee=fee,
                    pending_qty=pending_qty,
                    maker=maker,
                    slippage_bps=None if slippage_bps is None else float(slippage_bps),
                )
                entry_side = side_norm or entry.get("side") if entry else side_norm
                if entry_side:
                    _update_account_pending(symbol, str(entry_side))
                else:
                    _sync_locked_total()

            if status in {"canceled", "cancelled", "expired"} and symbol:
                if order_id and order is not None:
                    try:
                        remain_val = pending_qty
                        if remain_val is None:
                            remain_val = max(float(getattr(order, "pending_qty", 0.0) or 0.0), 0.0)
                        order_price = base_price if base_price is not None else exec_price
                        price_hint = res_dict.get("price")
                        if order_price is None and isinstance(price_hint, (int, float)):
                            order_price = float(price_hint)
                        side_for_event = side if side is not None else getattr(order, "side", None)
                        if order_price is not None and side_for_event is not None:
                            order_manager.ensure_order_event(
                                symbol=symbol,
                                order_id=order_id,
                                side=str(side_for_event),
                                price=float(order_price),
                                orig_qty=float(getattr(order, "qty", 0.0) or 0.0),
                                remaining_qty=float(remain_val or 0.0),
                            )
                    except Exception:
                        pass
                entry = order_manager.on_cancel(
                    symbol=symbol,
                    order_id=order_id,
                    reason=res_dict.get("reason"),
                )
                cancel_side = side_norm
                if entry and not cancel_side:
                    cancel_side = entry.get("side")
                if cancel_side:
                    _update_account_pending(symbol, str(cancel_side))
                else:
                    _sync_locked_total()

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
            if trade and abs(pos_qty) > risk.min_order_qty:
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
                    if qty_close <= 0:
                        log.info(
                            "orden submínima: qty %.8f por debajo de los mínimos",
                            abs(pos_qty),
                        )
                        SKIPS.inc()
                        order_manager.emit_skip("min_notional")
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
                    if abs(pending_qty) <= 1e-9:
                        pending_qty = 0.0
                    exec_price = float(resp.get("price", price))
                    if status == "rejected":
                        if resp.get("reason") == "insufficient_cash":
                            SKIPS.inc()
                            order_manager.emit_skip("insufficient_cash")
                        continue
                    order_id_val = resp.get("order_id") or resp.get("client_order_id")
                    order_id = str(order_id_val) if order_id_val is not None else None
                    if status not in {"rejected", "error"}:
                        order_manager.on_order(
                            symbol=symbol,
                            order_id=order_id,
                            side=close_side,
                            price=price,
                            orig_qty=qty_close,
                            remaining_qty=pending_qty,
                            pnl=float(getattr(broker.state, "realized_pnl", 0.0)),
                        )
                        _update_account_pending(symbol, close_side)
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
                        maker = bool(exec_price == price)
                        fee_bps = (
                            exec_broker.maker_fee_bps if maker else broker.taker_fee_bps
                        )
                        fee = filled_qty * exec_price * (fee_bps / 10000.0)
                        slippage = None
                        if price:
                            slippage = ((exec_price - price) / price) * 10000.0
                        update_position = getattr(risk.account, "update_position", None)
                        if callable(update_position):
                            direction = 1.0 if close_side == "buy" else -1.0
                            update_position(symbol, direction * filled_qty, price=exec_price)
                        order_manager.on_fill(
                            symbol=symbol,
                            order_id=order_id,
                            side=close_side,
                            fill_qty=filled_qty,
                            price=exec_price,
                            fee=fee,
                            pending_qty=pending_qty,
                            maker=maker,
                            slippage_bps=slippage,
                            pnl=delta_rpnl,
                        )
                        _update_account_pending(symbol, close_side)
                        cur_qty = risk.account.current_exposure(symbol)[0]
                        if step_size > 0 and abs(cur_qty) < step_size:
                            risk.account.positions[symbol] = 0.0
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
                    if abs(delta_qty) > risk.min_order_qty:
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
                        qty_scale = adjust_qty(
                            qty_scale, price, min_notional, step_size, risk.min_order_qty
                        )
                        if qty_scale <= 0:
                            log.info(
                                "orden submínima: qty %.8f por debajo de los mínimos",
                                abs(delta_qty),
                            )
                            SKIPS.inc()
                            order_manager.emit_skip("min_notional")
                            continue
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
                        if abs(pending_qty) <= 1e-9:
                            pending_qty = 0.0
                        exec_price = float(resp.get("price", price))
                        if status == "rejected":
                            if resp.get("reason") == "insufficient_cash":
                                SKIPS.inc()
                                order_manager.emit_skip("insufficient_cash")
                            continue
                        order_id_val = resp.get("order_id") or resp.get("client_order_id")
                        order_id = str(order_id_val) if order_id_val is not None else None
                        if status not in {"rejected", "error"}:
                            order_manager.on_order(
                                symbol=symbol,
                                order_id=order_id,
                                side=side,
                                price=price,
                                orig_qty=qty_scale,
                                remaining_qty=pending_qty,
                                pnl=float(getattr(broker.state, "realized_pnl", 0.0)),
                            )
                            _update_account_pending(symbol, side)
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
                            maker = bool(exec_price == price)
                            fee_bps = (
                                exec_broker.maker_fee_bps if maker else broker.taker_fee_bps
                            )
                            fee = filled_qty * exec_price * (fee_bps / 10000.0)
                            slippage = None
                            if price:
                                slippage = ((exec_price - price) / price) * 10000.0
                            update_position = getattr(risk.account, "update_position", None)
                            if callable(update_position):
                                direction = 1.0 if side == "buy" else -1.0
                                update_position(symbol, direction * filled_qty, price=exec_price)
                            order_manager.on_fill(
                                symbol=symbol,
                                order_id=order_id,
                                side=side,
                                fill_qty=filled_qty,
                                price=exec_price,
                                fee=fee,
                                pending_qty=pending_qty,
                                maker=maker,
                                slippage_bps=slippage,
                                pnl=delta_rpnl,
                            )
                            _update_account_pending(symbol, side)
                            cur_qty = risk.account.current_exposure(symbol)[0]
                            if step_size > 0 and abs(cur_qty) < step_size:
                                risk.account.positions[symbol] = 0.0
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
                        "orden submínima: qty %.8f por debajo de los mínimos", abs(delta)
                    )
                    SKIPS.inc()
                    order_manager.emit_skip("min_notional")
                else:
                    log.warning("orden bloqueada: %s", reason)
                    SKIPS.inc()
                    log.info(
                        "METRICS %s",
                        json.dumps({"event": "risk_check_reject", "reason": reason}),
                    )
                    order_manager.emit_skip(str(reason))
                continue
            side = "buy" if delta > 0 else "sell"
            price = getattr(signal, "limit_price", None)
            price = (
                price
                if price is not None
                else limit_price_from_close(side, closed.c, tick_size)
            )
            qty = adjust_qty(
                abs(delta), price, min_notional, step_size, risk.min_order_qty
            )
            if qty <= 0:
                log.info(
                    "orden submínima: qty %.8f por debajo de los mínimos", abs(delta)
                )
                SKIPS.inc()
                order_manager.emit_skip("min_notional")
                continue
            notional = qty * price
            if qty < step_size or notional < min_notional:
                log.info(
                    "orden submínima: qty %.8f notional %.8f por debajo de los mínimos",
                    qty,
                    notional,
                )
                order_manager.emit_skip("min_notional")
                continue
            if not risk.register_order(symbol, notional):
                reason = getattr(risk, "last_kill_reason", "register_reject")
                log.warning("registro de orden bloqueado: %s", reason)
                SKIPS.inc()
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "register_order_reject", "reason": reason}),
                )
                order_manager.emit_skip(reason)
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
            if abs(pending_qty) <= 1e-9:
                pending_qty = 0.0
            exec_price = float(resp.get("price", price))
            if status == "rejected":
                if resp.get("reason") == "insufficient_cash":
                    SKIPS.inc()
                    order_manager.emit_skip("insufficient_cash")
                continue
            order_id_val = resp.get("order_id") or resp.get("client_order_id")
            order_id = str(order_id_val) if order_id_val is not None else None
            if status not in {"rejected", "error"}:
                order_manager.on_order(
                    symbol=symbol,
                    order_id=order_id,
                    side=side,
                    price=price,
                    orig_qty=qty,
                    remaining_qty=pending_qty,
                    pnl=float(getattr(broker.state, "realized_pnl", 0.0)),
                )
                _update_account_pending(symbol, side)
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
                maker = bool(exec_price == price)
                fee_bps = exec_broker.maker_fee_bps if maker else broker.taker_fee_bps
                fee = filled_qty * exec_price * (fee_bps / 10000.0)
                slippage = None
                if price:
                    slippage = ((exec_price - price) / price) * 10000.0
                update_position = getattr(risk.account, "update_position", None)
                if callable(update_position):
                    direction = 1.0 if side == "buy" else -1.0
                    update_position(symbol, direction * filled_qty, price=exec_price)
                order_manager.on_fill(
                    symbol=symbol,
                    order_id=order_id,
                    side=side,
                    fill_qty=filled_qty,
                    price=exec_price,
                    fee=fee,
                    pending_qty=pending_qty,
                    maker=maker,
                    slippage_bps=slippage,
                    pnl=delta_rpnl,
                )
                _update_account_pending(symbol, side)
                cur_qty = risk.account.current_exposure(symbol)[0]
                if step_size > 0 and abs(cur_qty) < step_size:
                    risk.account.positions[symbol] = 0.0
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
