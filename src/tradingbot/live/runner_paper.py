from __future__ import annotations
import asyncio
import errno
import logging
from datetime import datetime, timezone
import time
import contextlib
import json
import uvicorn

from sqlalchemy.exc import OperationalError

from .runner import BarAggregator
from ..adapters.binance import BinanceWSAdapter
from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.bybit_spot import BybitSpotAdapter as BybitSpotWSAdapter
from ..adapters.bybit_futures import BybitFuturesAdapter
from ..adapters.okx_spot import OKXSpotAdapter as OKXSpotWSAdapter
from ..adapters.okx_futures import OKXFuturesAdapter
from ..execution.paper import PaperAdapter
from ..execution.router import ExecutionRouter
from ..utils.metrics import MARKET_LATENCY, AGG_COMPLETED
from ..broker.broker import Broker
from ..config import settings
from ..risk.service import load_positions
from ..risk.portfolio_guard import GuardConfig, PortfolioGuard
from ..risk.service import RiskService
from ..risk.correlation_service import CorrelationService
from ..strategies import STRATEGIES
from monitoring import panel

try:
    from ..storage.timescale import get_engine

    _CAN_PG = True
except Exception:  # pragma: no cover
    _CAN_PG = False

log = logging.getLogger(__name__)


WS_ADAPTERS = {
    ("binance", "spot"): BinanceSpotWSAdapter,
    ("binance", "futures"): BinanceWSAdapter,
    ("bybit", "spot"): BybitSpotWSAdapter,
    ("bybit", "futures"): BybitFuturesAdapter,
    ("okx", "spot"): OKXSpotWSAdapter,
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
) -> None:
    """Run a simple live pipeline entirely in paper mode."""
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
    broker = PaperAdapter()
    broker.state.cash = initial_cash
    if hasattr(broker.account, "update_cash"):
        broker.account.update_cash(initial_cash)

    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=0.5, venue="paper")
    )
    guard.refresh_usd_caps(initial_cash)
    corr = CorrelationService()
    risk = RiskService(
        guard,
        corr_service=corr,
        account=broker.account,
        risk_pct=risk_pct,
    )
    risk.allow_short = market != "spot"
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
    params = params or {}
    strat = (
        strat_cls(config_path=config_path, **params)
        if (config_path or params)
        else strat_cls()
    )
    strat.risk_service = risk

    router = ExecutionRouter(
        [broker],
        prefer="maker",
        on_partial_fill=strat.on_partial_fill,
        on_order_expiry=strat.on_order_expiry,
    )
    exec_broker = Broker(router)

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

    try:
        agg = BarAggregator(timeframe=timeframe)
    except TypeError:
        agg = BarAggregator()
    tick = getattr(settings, "tick_size", 0.0)
    purge_interval = settings.risk_purge_minutes * 60.0
    last_purge = time.time()

    def _limit_price(side: str) -> float:
        book = getattr(broker.state, "order_book", {}).get(symbol, {})
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        last = broker.state.last_px.get(symbol, 0.0)
        best_bid = bids[0][0] if bids else last
        best_ask = asks[0][0] if asks else last
        return (best_ask - tick) if side == "buy" else (best_bid + tick)

    last_progress = -1
    last_log = 0

    try:
        async for t in adapter.stream_trades(symbol):
            ts = t.get("ts") or datetime.now(timezone.utc)
            px = float(t.get("price"))
            qty = float(t.get("qty", 0.0))
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
            broker.update_last_price(symbol, px)
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
                    prev_rpnl = broker.state.realized_pnl
                    price = _limit_price(close_side)
                    log.info(
                        "METRICS %s",
                        json.dumps(
                            {
                                "event": "order",
                                "side": close_side,
                                "price": price,
                                "qty": abs(pos_qty),
                                "fee": 0.0,
                                "pnl": broker.state.realized_pnl,
                            }
                        ),
                    )
                    resp = await exec_broker.place_limit(
                        symbol,
                        close_side,
                        price,
                        abs(pos_qty),
                        on_partial_fill=strat.on_partial_fill,
                        on_order_expiry=strat.on_order_expiry,
                    )
                    filled_qty = float(resp.get("filled_qty", 0.0))
                    pending_qty = float(resp.get("pending_qty", 0.0))
                    risk.account.update_open_order(
                        symbol, close_side, filled_qty + pending_qty
                    )
                    risk.on_fill(symbol, close_side, filled_qty, venue="paper")
                    delta_rpnl = (
                        resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                    )
                    if filled_qty > 0:
                        exec_price = float(resp.get("price", price))
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
                                }
                            ),
                        )
                    delta_rpnl = (
                        resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
                    )
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
                        prev_rpnl = broker.state.realized_pnl
                        price = _limit_price(side)
                        log.info(
                            "METRICS %s",
                            json.dumps(
                                {
                                    "event": "order",
                                    "side": side,
                                    "price": price,
                                    "qty": abs(delta_qty),
                                    "fee": 0.0,
                                    "pnl": broker.state.realized_pnl,
                                }
                            ),
                        )
                        resp = await exec_broker.place_limit(
                            symbol,
                            side,
                            price,
                            abs(delta_qty),
                            on_partial_fill=strat.on_partial_fill,
                            on_order_expiry=strat.on_order_expiry,
                        )
                        filled_qty = float(resp.get("filled_qty", 0.0))
                        pending_qty = float(resp.get("pending_qty", 0.0))
                        risk.account.update_open_order(
                            symbol, side, filled_qty + pending_qty
                        )
                        risk.on_fill(symbol, side, filled_qty, venue="paper")
                        delta_rpnl = (
                            resp.get("realized_pnl", broker.state.realized_pnl)
                            - prev_rpnl
                        )
                        if filled_qty > 0:
                            exec_price = float(resp.get("price", price))
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
            MARKET_LATENCY.observe(latency)
            AGG_COMPLETED.set(len(agg.completed))
            log.debug("bars accumulated=%d", len(agg.completed))
            warmup_total = 140
            bars = len(agg.completed)
            if bars < warmup_total and bars % 10 == 0:
                log.info("Warm-up progress %d/%d", bars, warmup_total)
            if closed is None:
                continue
            correlations = await asyncio.to_thread(corr.get_correlations)
            risk.update_correlation(correlations, corr_threshold)
            df = await asyncio.to_thread(agg.last_n_bars_df, 200)
            progress = len(df)
            now = time.monotonic()
            if progress < 140:
                if progress != last_progress or now - last_log >= 5:
                    log.info("Warm-up progress %d/140", progress)
                    last_progress, last_log = progress, now
                continue
            bar = {"window": df, "symbol": symbol}
            signal = strat.on_bar(bar)
            if signal is None:
                continue
            signal_ts = getattr(signal, "signal_ts", time.time())
            allowed, reason, delta = risk.check_order(
                symbol,
                signal.side,
                closed.c,
                strength=signal.strength,
                volatility=bar.get("atr") or bar.get("volatility"),
                target_volatility=bar.get("target_volatility"),
            )
            if not allowed:
                log.warning("orden bloqueada: %s", reason)
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "risk_check_reject", "reason": reason}),
                )
                continue
            if abs(delta) <= 0:
                continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            price = getattr(signal, "limit_price", None)
            price = price if price is not None else _limit_price(side)
            notional = qty * price
            if not risk.register_order(symbol, notional):
                reason = getattr(risk, "last_kill_reason", "register_reject")
                log.warning("registro de orden bloqueado: %s", reason)
                log.info(
                    "METRICS %s",
                    json.dumps({"event": "register_order_reject", "reason": reason}),
                )
                continue
            prev_rpnl = broker.state.realized_pnl
            log.info(
                "METRICS %s",
                json.dumps(
                    {
                        "event": "order",
                        "side": side,
                        "price": price,
                        "qty": qty,
                        "fee": 0.0,
                        "pnl": broker.state.realized_pnl,
                    }
                ),
            )
            resp = await exec_broker.place_limit(
                symbol,
                side,
                price,
                qty,
                on_partial_fill=strat.on_partial_fill,
                on_order_expiry=strat.on_order_expiry,
                signal_ts=signal_ts,
            )
            filled_qty = float(resp.get("filled_qty", 0.0))
            pending_qty = float(resp.get("pending_qty", 0.0))
            risk.account.update_open_order(symbol, side, filled_qty + pending_qty)
            risk.on_fill(symbol, side, filled_qty, venue="paper")
            delta_rpnl = resp.get("realized_pnl", broker.state.realized_pnl) - prev_rpnl
            if filled_qty > 0:
                exec_price = float(resp.get("price", price))
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
                        }
                    ),
                )
            halted, reason = risk.daily_mark(broker, symbol, px, delta_rpnl)
            if halted:
                log.error("[HALT] motivo=%s", reason)
                break
    finally:
        server.should_exit = True
        if metrics_task is not None:
            metrics_task.cancel()
            with contextlib.suppress(Exception):
                await metrics_task
