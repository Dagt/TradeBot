from abc import ABC, abstractmethod
from typing import AsyncIterator
import asyncio
import contextlib
import logging
import time
import random
import json

import websockets

from ..core.symbols import normalize as normalize_symbol
from ..core.account import Account
from ..utils.metrics import WS_FAILURES, WS_RECONNECTS
from ..config import settings


class AdapterState:
    """Lightweight container tracking market state for an adapter."""

    def __init__(self) -> None:
        self.order_book: dict[str, dict] = {}
        self.last_px: dict[str, float] = {}


class ExchangeAdapter(ABC):
    """Base class for all exchange adapters.

    Provides basic request rate limiting, error logging and payload
    normalization helpers so individual adapters can remain small.
    """

    name: str

    def __init__(self, rate_limit_per_sec: float = 5.0) -> None:
        self._rate_limit_per_sec = rate_limit_per_sec
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self.log = logging.getLogger(getattr(self, "name", self.__class__.__name__))
        # live market data populated by streaming helpers
        self.state = AdapterState()
        # basic account information for the adapter
        self.account = Account(float("inf"))
        name = getattr(self, "name", "")
        self.ping_interval = getattr(
            settings, f"{name}_ping_interval", settings.adapter_ping_interval
        )
        self.max_backoff = getattr(
            settings, f"{name}_max_backoff", settings.adapter_max_backoff
        )
        self._fee_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Rate limiting helpers
    async def _throttle(self) -> None:
        """Ensure a minimum delay between REST requests."""
        async with self._lock:
            now = time.monotonic()
            wait = 1.0 / self._rate_limit_per_sec - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    async def _request(self, fn, *args, **kwargs):
        """Execute ``fn`` respecting rate limits and awaiting coroutines.

        ``ccxt.async_support`` exposes coroutine based HTTP methods.  The
        previous implementation offloaded synchronous calls to a thread via
        :func:`asyncio.to_thread`.  After migrating adapters to the async
        variant we simply await the result if it is awaitable, falling back to
        returning the value directly for backwards compatibility (used in
        tests with dummy synchronous functions).
        """

        await self._throttle()
        try:
            res = fn(*args, **kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res
        except Exception as e:  # pragma: no cover - logging only
            self.log.error("%s request failed: %s", self.name, e)
            raise

    async def close(self) -> None:
        """Close underlying REST client if it exposes ``close``."""
        rest = getattr(self, "rest", None)
        if rest and hasattr(rest, "close"):
            try:
                res = rest.close()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:  # pragma: no cover - best effort
                self.log.warning("%s close failed: %s", self.name, e)

        task = getattr(self, "_fee_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _start_fee_updater(self, interval: float = 3600.0) -> None:
        """Launch background task refreshing trading fees periodically."""

        if not hasattr(self, "update_fees"):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no running loop
            return

        async def _runner():
            while True:
                try:
                    await self.update_fees()  # type: ignore[attr-defined]
                except Exception as e:  # pragma: no cover - best effort
                    self.log.warning("update_fees failed: %s", e)
                await asyncio.sleep(interval)

        self._fee_task = loop.create_task(_runner())

    # ------------------------------------------------------------------
    # Normalization helpers
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalise ``symbol`` using :func:`tradingbot.core.symbols.normalize`.

        This wrapper maintains backwards compatibility with older code
        invoking :meth:`normalize_symbol` on adapters directly while the
        actual implementation lives in :mod:`tradingbot.core.symbols`.
        """
        return normalize_symbol(symbol)

    def _normalize(self, symbol: str) -> str:
        """Adapter specific normalisation helper.

        ``stream_*`` websocket methods historically relied on the module level
        :func:`tradingbot.core.symbols.normalize` which yields the internal
        representation (e.g. ``BTCUSDT``).  Certain venues such as OKX expect
        dashes between base and quote (``BTC-USDT``) when subscribing to
        channels.  This private helper routes all normalisation through
        :meth:`normalize_symbol` so adapters can override that method for venue
        specific behaviour while keeping a consistent ``_normalize`` entry
        point for streaming helpers.
        """

        return self.normalize_symbol(symbol)

    def normalize_trade(self, symbol, ts, price, qty, side) -> dict:
        return {"symbol": symbol, "ts": ts, "price": price, "qty": qty, "side": side}

    def normalize_order_book(self, symbol, ts, bids, asks) -> dict:
        """Return a flattened order book snapshot.

        Parameters
        ----------
        symbol: str
            Market symbol of the snapshot.
        ts: datetime
            Timestamp of the book.
        bids, asks: list[list[float, float]]
            Price/quantity tuples as provided by the venue.

        The result splits the price and quantity levels into separate lists
        so downstream code can persist them without additional conversions.
        """

        bid_px = [float(p) for p, _ in bids]
        bid_qty = [float(q) for _, q in bids]
        ask_px = [float(p) for p, _ in asks]
        ask_qty = [float(q) for _, q in asks]

        return {
            "symbol": symbol,
            "ts": ts,
            "bid_px": bid_px,
            "bid_qty": bid_qty,
            "ask_px": ask_px,
            "ask_qty": ask_qty,
        }

    # ------------------------------------------------------------------
    # Websocket helpers
    async def _ping(self, ws) -> None:
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await ws.ping()
            except asyncio.CancelledError:  # pragma: no cover - task cancelled
                break
            except Exception:  # pragma: no cover - network issues
                break

    async def _ws_messages(self, url: str, subscribe: str | None = None) -> AsyncIterator[str]:
        backoff = 1.0
        successes = 0
        while True:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    if subscribe:
                        await ws.send(subscribe)
                    successes += 1
                    if successes >= 3:
                        backoff = 1.0
                        successes = 0
                    ping_task = asyncio.create_task(self._ping(ws))
                    try:
                        while True:
                            msg = await ws.recv()
                            try:
                                data = json.loads(msg)
                            except Exception:
                                yield msg
                                continue
                            if data.get("method") == "heartbeat":
                                await ws.send(
                                    json.dumps(
                                        {
                                            "jsonrpc": "2.0",
                                            "id": 0,
                                            "method": "public/respond-heartbeat",
                                        }
                                    )
                                )
                                continue
                            yield msg
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await ping_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                WS_FAILURES.labels(adapter=self.name).inc()
                WS_RECONNECTS.labels(adapter=self.name).inc()
                successes = 0
                delay = backoff * random.uniform(0.5, 1.5)
                self.log.warning(
                    "WS disconnected (%s). Reconnecting in %.1fs ...", e, delay
                )
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, self.max_backoff)

    # ------------------------------------------------------------------
    # Abstract API
    @abstractmethod
    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Yield dicts with ts, price, qty, side."""

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        """Yield order book snapshots.

        Default implementation raises ``NotImplementedError`` to keep the
        interface optional for adapters that do not support order book data.
        """
        raise NotImplementedError

    async def fetch_funding(self, symbol: str):
        """Return funding information for ``symbol``.

        Adapters may override this; by default it is unsupported.
        """
        raise NotImplementedError

    async def fetch_basis(self, symbol: str):
        """Return basis information for ``symbol``.

        Adapters may override this; by default it is unsupported.
        """
        raise NotImplementedError

    async def fetch_oi(self, symbol: str):
        """Return open interest information for ``symbol``.

        Adapters may override this; by default it is unsupported.
        """
        raise NotImplementedError

    async def fetch_open_interest(self, symbol: str):
        """Alias of :meth:`fetch_oi` for clarity."""
        return await self.fetch_oi(symbol)

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """Return provider response (paper/live)."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> dict:
        ...
