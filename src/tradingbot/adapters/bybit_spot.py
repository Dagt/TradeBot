# src/tradingbot/adapters/bybit_spot.py
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import websockets

try:
    import ccxt
except Exception:  # pragma: no cover - ccxt optional during tests
    ccxt = None

from .base import ExchangeAdapter

log = logging.getLogger(__name__)


class BybitSpotAdapter(ExchangeAdapter):
    """Adapter para Spot de Bybit usando WS público y REST via CCXT."""

    name = "bybit_spot"

    def __init__(self):
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")
        # enableRateLimit respeta límites de la API
        self.rest = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "spot"}})

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://stream.bybit.com/v5/public/spot"
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"publicTrade.{sym}"]}
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        for t in msg.get("data", []) or []:
                            price = float(t.get("p"))
                            qty = float(t.get("v", 0))
                            side = t.get("S", "").lower()
                            ts = datetime.fromtimestamp(int(t.get("T", 0)) / 1000, tz=timezone.utc)
                            yield self.normalize_trade(symbol, ts, price, qty, side)
            except Exception as e:  # pragma: no cover - network paths
                log.warning("Bybit spot trade WS error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://stream.bybit.com/v5/public/spot"
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"orderbook.1.{sym}"]}
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = msg.get("data") or {}
                        if not data:
                            continue
                        bids = [[float(p), float(q)] for p, q, *_ in data.get("b", [])]
                        asks = [[float(p), float(q)] for p, q, *_ in data.get("a", [])]
                        ts = datetime.fromtimestamp(int(data.get("ts", 0)) / 1000, tz=timezone.utc)
                        yield self.normalize_order_book(symbol, ts, bids, asks)
            except Exception as e:  # pragma: no cover - network paths
                log.warning("Bybit spot book WS error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:
            raise NotImplementedError("Funding not supported")
        return await self._request(method, sym)

    async def fetch_oi(self, symbol: str):
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "fetchOpenInterest", None)
        if method:
            return await self._request(method, sym)
        hist = getattr(self.rest, "fetchOpenInterestHistory", None)
        if hist:
            data = await self._request(hist, sym)
            return data[-1] if isinstance(data, list) and data else data
        raise NotImplementedError("Open interest not supported")

    async def place_order(self, symbol: str, side: str, type_: str, qty: float,
                          price: float | None = None) -> dict:
        return await self._to_thread(self.rest.create_order, symbol, type_, side, qty, price)

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        return await self._to_thread(self.rest.cancel_order, order_id, symbol)
