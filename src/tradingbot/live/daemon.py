from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Dict, List, Optional

from ..bus import EventBus
from ..connectors.base import ExchangeConnector
from ..execution.order_types import Order
from ..execution.router import ExecutionRouter
from ..risk.manager import RiskManager

log = logging.getLogger(__name__)


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
        strategies: Iterable,
        risk_manager: RiskManager,
        router: ExecutionRouter,
        symbols: Iterable[str],
        accounts: Optional[Dict[str, ExchangeConnector]] = None,
    ) -> None:
        self.adapters = adapters
        self.strategies = list(strategies)
        self.risk = risk_manager
        self.router = router
        self.symbols = list(symbols)
        self.accounts = accounts or adapters

        # Event bus used across components
        self.bus = risk_manager.bus or EventBus()
        if risk_manager.bus is None:
            risk_manager.bus = self.bus

        self._tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event()
        self.balances: Dict[str, dict] = {}

        # Bus subscriptions
        self.bus.subscribe("trade", self._dispatch_trade)
        self.bus.subscribe("signal", self._on_signal)
        self.bus.subscribe("risk:halted", self._on_halt)

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
            except Exception as exc:  # pragma: no cover - network
                log.warning("balance_error", extra={"account": name, "err": str(exc)})

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

    # ------------------------------------------------------------------
    async def _on_signal(self, evt: dict) -> None:
        """Size and route signals coming from strategies."""
        signal = evt.get("signal")
        trade = evt.get("trade")
        symbol = getattr(trade, "symbol", None)
        price = getattr(trade, "price", None)
        side = getattr(signal, "side", "")
        strength = getattr(signal, "strength", 1.0)

        delta = self.risk.size(side, strength)
        if abs(delta) <= 0:
            return
        if price is not None and not self.risk.check_limits(price):
            log.warning("risk_halt", extra={"price": price})
            return
        order = Order(symbol=symbol, side="buy" if delta > 0 else "sell", type_="market", qty=abs(delta))
        res = await self.router.execute(order)
        await self.bus.publish("fill", {"symbol": symbol, "side": order.side, "qty": order.qty, **res})

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
        for name, adapter in self.adapters.items():
            for symbol in self.symbols:
                task = asyncio.create_task(self._stream_adapter(name, adapter, symbol))
                self._tasks.append(task)
        await self._stop.wait()
        await asyncio.gather(*self._tasks, return_exceptions=True)
