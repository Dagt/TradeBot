from __future__ import annotations
import asyncio
import errno
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Type, List, Any
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
    log.info(
        "Connecting to %s %s testnet for %s", exchange, market, symbol
    )
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
    dguard = DailyGuard(GuardLimits(
        daily_max_loss_pct=daily_max_loss_pct,
        daily_max_drawdown_pct=daily_max_drawdown_pct,
        halt_action="close_all",
    ), venue=venue)
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
        if not isinstance(res, dict):
            return
        status = str(res.get("status", "")).lower()
        if status not in {"canceled", "cancelled", "expired"}:
            return
        if res.get("_cancel_handled"):
            return
        res["_cancel_handled"] = True
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
        locked = risk.account.get_locked_usd(symbol) if symbol else 0.0
        log.info(
            "METRICS %s",
            json.dumps(
                {"event": "cancel", "reason": res.get("reason"), "locked": locked}
            ),
        )
        metric_pending = res.get("pending_qty", pending_qty)
        if metric_pending_override is not None:
            metric_pending = metric_pending_override
        try:
            metric_pending = float(metric_pending)
        except (TypeError, ValueError):
            metric_pending = 0.0
        if metric_pending <= 0:
            return  # treat as filled; no cancel handling needed
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
                    get_locked = getattr(risk.account, "get_locked_usd", None)
                    locked = 0.0
                    if callable(get_locked):
                        try:
                            locked = float(get_locked(symbol))
                        except Exception:
                            locked = 0.0
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
            if action in {"re_quote", "requote", "re-quote"}:
                return None
            return action
        return _cb

    on_pf = _wrap_cb(getattr(strat, "on_partial_fill", None))
    on_oe = _wrap_cb(getattr(strat, "on_order_expiry", None), call_cancel=True)
    tif = f"GTD:{expiry}|PO"
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
        last_price = px
        broker.update_last_price(symbol, px)
        risk.mark_price(symbol, px)
        risk.update_correlation(corr.get_correlations(), corr_threshold)
        halted, reason = risk.daily_mark(broker, symbol, px, 0.0)
        if halted:
            log.error("[HALT] motivo=%s", reason)
            break
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
        pending = risk.account.open_orders.get(symbol, {}).get(sig.side, 0.0)
        allowed, reason, delta = risk.check_order(
            symbol,
            sig.side,
            closed.c,
            strength=sig.strength,
            corr_threshold=corr_threshold,
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
            locked = risk.account.get_locked_usd(symbol)
            log.info(
                "METRICS %s",
                json.dumps({"exposure": cur_qty, "locked": locked}),
            )
        risk.on_fill(
            symbol, side, filled_qty, venue=venue if not dry_run else "paper"
        )
        delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
        halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
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
    *,
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
    """Run a simple live loop on a crypto exchange testnet.

    Parameters
    ----------
    slip_bps_per_qty:
        Optional manual slippage in basis points applied per unit of traded
        quantity. When omitted, slippage is automatically estimated from order
        book depth or historical executions.
    """
    log.info("Starting testnet runner for %s %s", exchange, market)
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
