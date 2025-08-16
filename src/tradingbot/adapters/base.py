from abc import ABC, abstractmethod
from typing import AsyncIterator
import asyncio
import logging
import time


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
        """Run ``fn`` in a thread respecting rate limits and logging errors."""
        await self._throttle()
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as e:  # pragma: no cover - logging only
            self.log.error("%s request failed: %s", self.name, e)
            raise

    # ------------------------------------------------------------------
    # Normalization helpers
    def normalize_symbol(self, symbol: str) -> str:
        """Default symbol normalisation removing separators."""
        return symbol.replace("/", "")

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

    async def fetch_oi(self, symbol: str):
        """Return open interest information for ``symbol``.

        Adapters may override this; by default it is unsupported.
        """
        raise NotImplementedError

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
    ) -> dict:
        """Return provider response (paper/live)."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> dict:
        ...
