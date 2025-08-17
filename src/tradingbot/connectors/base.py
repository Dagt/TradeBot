"""Common connector utilities and data models."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, List, Tuple

import ccxt.async_support as ccxt
import websockets
from pydantic import BaseModel


class Trade(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    price: float
    amount: float
    side: str


class OrderBook(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]


class Funding(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    rate: float


class Basis(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    basis: float


class OpenInterest(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    oi: float


class ExchangeConnector:
    """Base connector implementing REST helpers and WS streaming."""

    name: str = ""

    def __init__(self, rate_limit: int = 5) -> None:
        self.rest = getattr(ccxt, self.name)({"enableRateLimit": True})
        self._sem = asyncio.Semaphore(rate_limit)
        self.log = logging.getLogger(self.__class__.__name__)
        self.reconnect_delay = 1
        self.ping_interval = 20

    async def _rest_call(self, fn: Callable[..., Awaitable[Any]], *a, **k) -> Any:
        async with self._sem:
            try:
                return await fn(*a, **k)
            except Exception as e:  # pragma: no cover - logging only
                self.log.error("rest_error", extra={"err": str(e)})
                raise

    async def fetch_trades(self, symbol: str) -> List[Trade]:
        data = await self._rest_call(self.rest.fetch_trades, symbol)
        trades: List[Trade] = []
        for t in data:
            trades.append(
                Trade(
                    timestamp=datetime.fromtimestamp(t["timestamp"] / 1000),
                    exchange=self.name,
                    symbol=symbol,
                    price=float(t["price"]),
                    amount=float(t["amount"]),
                    side=t.get("side", ""),
                )
            )
        return trades

    async def fetch_funding(self, symbol: str) -> Funding:
        data = await self._rest_call(self.rest.fetch_funding_rate, symbol)
        ts = data.get("timestamp") or data.get("time") or 0
        if ts > 1e12:  # milliseconds
            ts /= 1000
        return Funding(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            rate=float(
                data.get("fundingRate")
                or data.get("rate")
                or data.get("value")
                or 0.0
            ),
        )

    async def fetch_basis(self, symbol: str) -> Basis:
        data = await self._rest_call(self.rest.fetch_basis, symbol)
        ts = data.get("timestamp") or data.get("time") or 0
        if ts > 1e12:
            ts /= 1000
        return Basis(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            basis=float(data.get("basis") or data.get("value") or 0.0),
        )

    async def fetch_open_interest(self, symbol: str) -> OpenInterest:
        data = await self._rest_call(self.rest.fetch_open_interest, symbol)
        ts = data.get("timestamp") or data.get("time") or 0
        if ts > 1e12:
            ts /= 1000
        return OpenInterest(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            oi=float(data.get("openInterest") or data.get("oi") or data.get("value") or 0.0),
        )

    async def fetch_oi(self, symbol: str) -> OpenInterest:
        """Alias for :meth:`fetch_open_interest` to maintain backwards compatibility."""
        return await self.fetch_open_interest(symbol)

    async def _ping(self, ws) -> None:
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await ws.ping()
            except asyncio.CancelledError:  # pragma: no cover - task cancelled
                break
            except Exception:  # pragma: no cover - network issues
                break

    async def stream_order_book(self, symbol: str):
        url = self._ws_url(symbol)
        subscribe = self._ws_subscribe(symbol)
        backoff = self.reconnect_delay
        while True:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    await ws.send(subscribe)
                    backoff = self.reconnect_delay
                    ping_task = asyncio.create_task(self._ping(ws))
                    try:
                        while True:
                            msg = await ws.recv()
                            yield self._parse_order_book(msg, symbol)
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(Exception):
                            await ping_task
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover - network/streaming
                self.log.warning("ws_reconnect", extra={"err": str(e)})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def stream_trades(self, symbol: str):
        """Stream trades for the given symbol."""
        url = self._ws_trades_url(symbol)
        subscribe = self._ws_trades_subscribe(symbol)
        backoff = self.reconnect_delay
        while True:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    await ws.send(subscribe)
                    backoff = self.reconnect_delay
                    ping_task = asyncio.create_task(self._ping(ws))
                    try:
                        while True:
                            msg = await ws.recv()
                            yield self._parse_trade(msg, symbol)
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(Exception):
                            await ping_task
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover - network/streaming
                self.log.warning("ws_reconnect", extra={"err": str(e)})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # --- methods to be implemented by subclasses ---
    def _ws_url(self, symbol: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _ws_subscribe(self, symbol: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:  # pragma: no cover - abstract
        raise NotImplementedError

    def _ws_trades_url(self, symbol: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _ws_trades_subscribe(self, symbol: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _parse_trade(self, msg: str, symbol: str) -> Trade:  # pragma: no cover - abstract
        raise NotImplementedError
