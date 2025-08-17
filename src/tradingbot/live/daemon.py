from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from importlib import import_module
from collections.abc import Iterable
from typing import Dict, List, Optional
import os
from pathlib import Path

import pandas as pd
import sentry_sdk
import yaml
from sentry_sdk import capture_exception
from ..data.features import returns

from ..bus import EventBus
from ..connectors.base import ExchangeConnector
from ..execution.order_types import Order
from ..execution.router import ExecutionRouter
from ..risk.manager import RiskManager
from ..risk.portfolio_guard import PortfolioGuard
from ..strategies.cross_exchange_arbitrage import CrossArbConfig
from ..data.funding import poll_funding
from ..data.open_interest import poll_open_interest
from ..data.basis import poll_basis
from ..execution.balance import rebalance_between_exchanges
from ..utils.metrics import BASIS, FUNDING_RATE, OPEN_INTEREST

log = logging.getLogger(__name__)


def _init_sentry() -> None:
    """Initialize Sentry from config file and environment variables."""
    cfg_path = Path(__file__).resolve().parents[3] / "monitoring" / "sentry.yml"
    config: dict = {}
    try:
        with cfg_path.open("r") as fh:
            config = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        config = {}

    dsn = os.getenv("SENTRY_DSN", config.get("dsn"))
    enabled = os.getenv("SENTRY_ENABLED")
    if enabled is not None and enabled.lower() not in {"1", "true", "yes"}:
        return
    if not dsn:
        return

    init_kwargs = {
        "dsn": dsn,
        "environment": os.getenv("SENTRY_ENVIRONMENT", config.get("environment")),
    }
    sample_rate = os.getenv(
        "SENTRY_TRACES_SAMPLE_RATE", config.get("traces_sample_rate")
    )
    if sample_rate is not None:
        try:
            init_kwargs["traces_sample_rate"] = float(sample_rate)
        except (TypeError, ValueError):
            pass
    sentry_sdk.init(**{k: v for k, v in init_kwargs.items() if v is not None})


_init_sentry()


def _report_task_error(task: asyncio.Task) -> None:
    """Send unhandled task exceptions to Sentry."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:  # pragma: no cover - task management
        return
    if exc:
        log.exception("task_error", exc_info=exc)
        capture_exception(exc)


def _load_obj(path: str):
    """Import and return an object given ``module.object`` path."""
    module, _, name = path.rpartition(".")
    mod = import_module(module)
    return getattr(mod, name)


class TradeBotDaemon:
    """Simple orchestrator coordinating market data, strategies and execution.

    Parameters
    ----------
    adapters:
        Mapping of venue name to data adapter/connector.  They must expose
        ``stream_trades`` and ``rest.fetch_balance``.
    strategies:
        Iterable of strategy instances.  Strategies are expected to expose
        ``on_trade`` returning an optional signal with ``side`` and
        ``strength`` attributes.
    risk_manager:
        Instance of :class:`RiskManager` used to gate orders.
    router:
        Execution router responsible for sending orders to the venues.
    symbols:
        List of symbols to subscribe on each adapter.
    accounts:
        Optional mapping of account name to connector for balance management.
    """

    def __init__(
        self,
        adapters: Dict[str, ExchangeConnector],
        strategies: Iterable | None,
        risk_manager: RiskManager,
        router: ExecutionRouter,
        symbols: Iterable[str],
        accounts: Optional[Dict[str, ExchangeConnector]] = None,
        *,
        guard: PortfolioGuard | None = None,
        cross_arbs: Iterable[CrossArbConfig] | None = None,
        strategy_paths: Iterable[str] | None = None,
        arb_runners: Iterable[tuple[str, dict]] | None = None,
        balance_interval: float = 60.0,
        correlation_threshold: float = 0.8,
        returns_window: int = 100,
        rebalance_assets: Iterable[str] | None = None,
        rebalance_threshold: float = 0.0,
        rebalance_interval: float | None = None,
        rebalance_enabled: bool = False,
    ) -> None:
        self.adapters = adapters
        self.strategies: List[object] = []
        if strategies:
            self.strategies.extend(strategies)
        for path in strategy_paths or []:
            try:
                cls = _load_obj(path)
                self.strategies.append(cls())
            except Exception as exc:
                log.warning("strategy_load_error", extra={"path": path, "err": str(exc)})
        self.risk = risk_manager
        self.router = router
        # ensure router knows about all adapters (paper/live)
        for name, ad in adapters.items():
            self.router.adapters.setdefault(name, ad)
        self.symbols = list(symbols)
        self.accounts = accounts or adapters
        self.guard = guard
        self.cross_arbs = list(cross_arbs or [])
        self.arb_runners: List[tuple] = []
        for path, kwargs in arb_runners or []:
            try:
                fn = _load_obj(path)
                self.arb_runners.append((fn, kwargs or {}))
            except Exception as exc:
                log.warning("runner_load_error", extra={"path": path, "err": str(exc)})
        self.balance_interval = balance_interval
        self.corr_threshold = float(correlation_threshold)
        self.rebalance_assets = list(rebalance_assets or [])
        self.rebalance_threshold = float(rebalance_threshold)
        self.rebalance_interval = (
            rebalance_interval if rebalance_interval is not None else balance_interval
        )
        self.rebalance_enabled = rebalance_enabled

        # initialize position books for all venues/symbols
        for venue in self.adapters:
            for sym in self.symbols:
                self.risk.update_position(venue, sym, 0.0)
                if self.guard:
                    self.guard.set_position(venue, sym, 0.0)

        # Event bus used across components
        self.bus = risk_manager.bus or EventBus()
        if risk_manager.bus is None:
            risk_manager.bus = self.bus

        self._tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event()
        self.balances: Dict[str, dict] = {}
        self.price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=returns_window)
        )
        # Últimos precios conocidos por símbolo/asset
        self.last_prices: Dict[str, float] = {}

        # Bus subscriptions
        self.bus.subscribe("trade", self._dispatch_trade)
        self.bus.subscribe("signal", self._on_signal)
        self.bus.subscribe("risk:halted", self._on_halt)
        self.bus.subscribe("control:halt", self._on_external_halt)
        self.bus.subscribe("capital:update", self._on_capital_update)
        self.bus.subscribe("funding", self._dispatch_funding)
        self.bus.subscribe("basis", self._dispatch_basis)
        self.bus.subscribe("open_interest", self._dispatch_open_interest)
        # Mantén marca de precios desde el bus
        self.bus.subscribe("price", self._on_price_event)

    # ------------------------------------------------------------------
    async def refresh_balances(self) -> None:
        """Fetch balances for all configured accounts."""
        for name, conn in self.accounts.items():
            fetch = getattr(getattr(conn, "rest", conn), "fetch_balance", None)
            if fetch is None:
                continue
            try:
                bal = await fetch()  # type: ignore[misc]
                self.balances[name] = bal
                log.info("balance", extra={"account": name, "data": bal})
                # sincroniza posición en RiskManager y PortfolioGuard
                for sym, info in bal.items():
                    try:
                        qty = float(info.get("total", info))  # soporta dict o float
                    except AttributeError:
                        qty = float(info)
                    self.risk.update_position(name, sym, qty)
                    if self.guard:
                        self.guard.set_position(name, sym, qty)
            except Exception as exc:  # pragma: no cover - network
                log.warning("balance_error", extra={"account": name, "err": str(exc)})
                capture_exception(exc)
        self._reconcile_balances()

    # ------------------------------------------------------------------
    def _reconcile_balances(self) -> None:
        """Log balance differences across venues."""
        venues = list(self.balances.keys())
        if len(venues) < 2:
            return
        assets = set()
        for bal in self.balances.values():
            assets.update(bal.keys())
        for asset in assets:
            vals = {v: float(self.balances[v].get(asset, 0.0)) for v in venues}
            if len(set(vals.values())) > 1:
                log.info("balance_reconcile", extra={"asset": asset, "balances": vals})

    # ------------------------------------------------------------------
    async def _on_price_event(self, evt: dict) -> None:
        """Store latest price marks coming from the bus."""
        symbol = evt.get("symbol") or evt.get("asset")
        price = evt.get("price")
        if symbol and price is not None:
            try:
                self.last_prices[symbol] = float(price)
            except (TypeError, ValueError):
                return
            if self.guard:
                try:
                    self.guard.mark_price(symbol, float(price))
                except Exception:  # pragma: no cover - defensive
                    pass

    # ------------------------------------------------------------------
    async def _balance_worker(self) -> None:
        while not self._stop.is_set():
            await self.refresh_balances()
            await asyncio.sleep(self.balance_interval)

    # ------------------------------------------------------------------
    async def _rebalance_worker(self) -> None:
        engine = getattr(self.router, "_engine", None)
        while not self._stop.is_set():
            for asset in self.rebalance_assets:
                try:
                    # Fetch reference price from available sources
                    price: float | None = None
                    risk_prices = getattr(self.risk, "prices", None)
                    if price is None and isinstance(risk_prices, dict):
                        price = risk_prices.get(asset)
                    if price is None and self.guard is not None:
                        price = self.guard.st.prices.get(asset)
                    if price is None:
                        price = self.last_prices.get(asset)
                    if price is None:
                        price = 1.0
                    await rebalance_between_exchanges(
                        asset,
                        price=float(price),
                        venues=self.accounts,
                        risk=self.risk,
                        engine=engine,
                        threshold=self.rebalance_threshold,
                    )
                except Exception as exc:  # pragma: no cover - network
                    log.warning(
                        "rebalance_error", extra={"asset": asset, "err": str(exc)}
                    )
                    capture_exception(exc)
            await asyncio.sleep(self.rebalance_interval)

    # ------------------------------------------------------------------
    async def _cross_arbitrage(self, cfg: CrossArbConfig) -> None:
        """Simplified cross exchange arbitrage loop."""
        last: Dict[str, Optional[float]] = {"spot": None, "perp": None}
        balances: Dict[str, float] = {cfg.spot.name: 0.0, cfg.perp.name: 0.0}
        lock = asyncio.Lock()

        async def maybe_trade() -> None:
            if last["spot"] is None or last["perp"] is None:
                return
            edge = (last["perp"] - last["spot"]) / last["spot"]
            if abs(edge) < cfg.threshold:
                return
            async with lock:
                edge = (last["perp"] - last["spot"]) / last["spot"]
                if abs(edge) < cfg.threshold:
                    return
                qty = cfg.notional / last["spot"]
                if edge > 0:
                    spot_side, perp_side = "buy", "sell"
                else:
                    spot_side, perp_side = "sell", "buy"
                resp_spot, resp_perp = await asyncio.gather(
                    cfg.spot.place_order(cfg.symbol, spot_side, "market", qty),
                    cfg.perp.place_order(cfg.symbol, perp_side, "market", qty),
                )
                balances[cfg.spot.name] += qty if spot_side == "buy" else -qty
                balances[cfg.perp.name] += qty if perp_side == "buy" else -qty
                self.risk.update_position(cfg.spot.name, cfg.symbol, balances[cfg.spot.name])
                self.risk.update_position(cfg.perp.name, cfg.symbol, balances[cfg.perp.name])
                if self.guard:
                    self.guard.update_position_on_order(cfg.symbol, spot_side, qty, cfg.spot.name)
                    self.guard.update_position_on_order(cfg.symbol, perp_side, qty, cfg.perp.name)
                pnl = (resp_perp.get("price", 0.0) - resp_spot.get("price", 0.0)) * qty
                self.risk.update_pnl(pnl, venue=cfg.perp.name)
                self.risk.update_pnl(-pnl, venue=cfg.spot.name)
                await self.bus.publish(
                    "fill",
                    {"symbol": cfg.symbol, "side": spot_side, "qty": qty, "venue": cfg.spot.name, **resp_spot},
                )
                await self.bus.publish(
                    "fill",
                    {"symbol": cfg.symbol, "side": perp_side, "qty": qty, "venue": cfg.perp.name, **resp_perp},
                )
                log.info(
                    "CROSS ARB edge=%.4f%% spot=%.2f perp=%.2f qty=%.6f pos_spot=%.6f pos_perp=%.6f",
                    edge * 100,
                    last["spot"],
                    last["perp"],
                    qty,
                    balances[cfg.spot.name],
                    balances[cfg.perp.name],
                )

        async def listen_spot() -> None:
            async for t in cfg.spot.stream_trades(cfg.symbol):
                px = t.get("price")
                if px is not None:
                    last["spot"] = float(px)
                    await maybe_trade()

        async def listen_perp() -> None:
            async for t in cfg.perp.stream_trades(cfg.symbol):
                px = t.get("price")
                if px is not None:
                    last["perp"] = float(px)
                    await maybe_trade()

        await asyncio.gather(listen_spot(), listen_perp())

    # ------------------------------------------------------------------
    async def _stream_adapter(self, name: str, adapter: ExchangeConnector, symbol: str) -> None:
        """Stream trades from an adapter with automatic reconnection."""
        while not self._stop.is_set():
            try:
                async for trade in adapter.stream_trades(symbol):
                    await self.bus.publish("trade", trade)
            except asyncio.CancelledError:  # pragma: no cover - task management
                raise
            except Exception as exc:  # pragma: no cover - network
                delay = getattr(adapter, "reconnect_delay", 1)
                log.warning("ws_reconnect", extra={"adapter": name, "err": str(exc)})
                capture_exception(exc)
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    async def _dispatch_trade(self, trade) -> None:
        """Forward trade events to strategies and publish resulting signals."""
        for strat in self.strategies:
            handler = getattr(strat, "on_trade", None)
            if handler is None:
                continue
            try:
                signal = handler(trade)
            except Exception as exc:  # pragma: no cover - strategy error
                log.exception("strategy_error", extra={"strategy": strat.__class__.__name__, "err": str(exc)})
                continue
            if signal is not None:
                await self.bus.publish("signal", {"signal": signal, "trade": trade})

    async def _dispatch_funding(self, evt: dict) -> None:
        """Forward funding events to strategies."""
        rate = float(evt.get("rate") or 0.0)
        symbol = evt.get("symbol", "")
        FUNDING_RATE.labels(symbol=symbol).set(rate)
        for strat in self.strategies:
            handler = getattr(strat, "on_funding", None)
            if handler is None:
                continue
            try:
                signal = handler(evt)
            except Exception as exc:  # pragma: no cover - strategy error
                log.exception("strategy_error", extra={"strategy": strat.__class__.__name__, "err": str(exc)})
                continue
            if signal is not None:
                await self.bus.publish("signal", {"signal": signal, "funding": evt})

    async def _dispatch_basis(self, evt: dict) -> None:
        """Forward basis events to strategies."""
        value = float(evt.get("basis") or 0.0)
        symbol = evt.get("symbol", "")
        BASIS.labels(symbol=symbol).set(value)
        for strat in self.strategies:
            handler = getattr(strat, "on_basis", None)
            if handler is None:
                continue
            try:
                signal = handler(evt)
            except Exception as exc:  # pragma: no cover - strategy error
                log.exception("strategy_error", extra={"strategy": strat.__class__.__name__, "err": str(exc)})
                continue
            if signal is not None:
                await self.bus.publish("signal", {"signal": signal, "basis": evt})

    async def _dispatch_open_interest(self, evt: dict) -> None:
        """Forward open interest events to strategies."""
        oi = float(evt.get("oi") or 0.0)
        symbol = evt.get("symbol", "")
        OPEN_INTEREST.labels(symbol=symbol).set(oi)
        for strat in self.strategies:
            handler = getattr(strat, "on_open_interest", None)
            if handler is None:
                continue
            try:
                signal = handler(evt)
            except Exception as exc:  # pragma: no cover - strategy error
                log.exception("strategy_error", extra={"strategy": strat.__class__.__name__, "err": str(exc)})
                continue
            if signal is not None:
                await self.bus.publish("signal", {"signal": signal, "open_interest": evt})

    # ------------------------------------------------------------------
    async def _on_signal(self, evt: dict) -> None:
        """Size and route signals coming from strategies."""
        signal = evt.get("signal")
        trade = evt.get("trade")
        symbol = getattr(trade, "symbol", None)
        price = getattr(trade, "price", None)
        side = getattr(signal, "side", "")
        strength = getattr(signal, "strength", 1.0)
        symbol_vol = 0.0
        if symbol and price is not None:
            hist = self.price_history[symbol]
            hist.append(float(price))
            if len(hist) >= 2:
                df_hist = pd.DataFrame({"close": list(hist)})
                rets = returns(df_hist).dropna()
                symbol_vol = float(rets.std()) if not rets.empty else 0.0
        returns_dict: Dict[str, List[float]] = {}
        for sym, hist in self.price_history.items():
            if len(hist) >= 2:
                df_hist = pd.DataFrame({"close": list(hist)})
                rets_list = returns(df_hist).dropna().tolist()
                if rets_list:
                    returns_dict[sym] = rets_list
        corr_pairs: Dict[tuple[str, str], float] = {}
        if returns_dict:
            cov_matrix = self.risk.covariance_matrix(returns_dict)
            vars: Dict[str, float] = {
                sym: abs(cov_matrix.get((sym, sym), 0.0)) for sym in returns_dict
            }
            for (a, b), cov in cov_matrix.items():
                if a == b:
                    continue
                va = vars.get(a)
                vb = vars.get(b)
                if va and vb and va > 0 and vb > 0:
                    corr_pairs[(a, b)] = cov / ((va ** 0.5) * (vb ** 0.5))
            self.risk.update_correlation(corr_pairs, self.corr_threshold)
            self.risk.update_covariance(cov_matrix, self.corr_threshold)

        delta = self.risk.size(side, strength)
        delta += self.risk.size_with_volatility(symbol_vol)
        delta = self.risk.adjust_size_for_correlation(
            symbol, delta, corr_pairs, self.corr_threshold
        )
        if abs(delta) <= 0:
            return
        if price is not None and not self.risk.check_limits(price):
            log.warning("risk_halt", extra={"price": price})
            return
        order = Order(symbol=symbol, side="buy" if delta > 0 else "sell", type_="market", qty=abs(delta))
        res = await self.router.execute(order)
        venue = res.get("venue")
        await self.bus.publish("fill", {"symbol": symbol, "side": order.side, "qty": order.qty, "venue": venue, **res})
        if venue and self.guard:
            self.guard.update_position_on_order(symbol, order.side, order.qty, venue)

    # ------------------------------------------------------------------
    async def _on_external_halt(self, evt: dict) -> None:
        """Receive external kill-switch and propagate through bus."""
        reason = evt.get("reason", "external")
        await self.bus.publish("risk:halted", {"reason": reason})

    # ------------------------------------------------------------------
    async def _on_capital_update(self, evt: dict) -> None:
        """Adjust portfolio guard limits from external controller."""
        if not self.guard:
            return
        tot = evt.get("total_cap_usdt")
        per = evt.get("per_symbol_cap_usdt")
        if tot is not None:
            self.guard.cfg.total_cap_usdt = float(tot)
        if per is not None:
            self.guard.cfg.per_symbol_cap_usdt = float(per)

    # ------------------------------------------------------------------
    async def _on_halt(self, _evt: dict) -> None:
        """Stop the daemon when risk manager triggers a halt."""
        self._stop.set()
        for t in self._tasks:
            t.cancel()

    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Entry point for running the daemon until halted."""
        await self.refresh_balances()
        # periodic balance reconciliation
        task = asyncio.create_task(self._balance_worker())
        task.add_done_callback(_report_task_error)
        self._tasks.append(task)
        if self.rebalance_enabled and self.rebalance_assets:
            task = asyncio.create_task(self._rebalance_worker())
            task.add_done_callback(_report_task_error)
            self._tasks.append(task)
        # cross exchange arbitrage loops
        for cfg in self.cross_arbs:
            task = asyncio.create_task(self._cross_arbitrage(cfg))
            task.add_done_callback(_report_task_error)
            self._tasks.append(task)
        # dynamic arbitrage runners
        for func, kwargs in self.arb_runners:
            task = asyncio.create_task(func(**kwargs))
            task.add_done_callback(_report_task_error)
            self._tasks.append(task)
        # market data streams
        for name, adapter in self.adapters.items():
            for symbol in self.symbols:
                task = asyncio.create_task(self._stream_adapter(name, adapter, symbol))
                task.add_done_callback(_report_task_error)
                self._tasks.append(task)
                if getattr(adapter, "fetch_funding", None):
                    task = asyncio.create_task(
                        poll_funding(adapter, symbol, self.bus, persist_pg=True)
                    )
                    task.add_done_callback(_report_task_error)
                    self._tasks.append(task)
                if getattr(adapter, "fetch_open_interest", None) or getattr(adapter, "fetch_oi", None):
                    task = asyncio.create_task(
                        poll_open_interest(adapter, symbol, self.bus, persist_pg=True)
                    )
                    task.add_done_callback(_report_task_error)
                    self._tasks.append(task)
                if getattr(adapter, "fetch_basis", None):
                    task = asyncio.create_task(
                        poll_basis(adapter, symbol, self.bus, persist_pg=True)
                    )
                    task.add_done_callback(_report_task_error)
                    self._tasks.append(task)
        await self._stop.wait()
        await asyncio.gather(*self._tasks, return_exceptions=True)
