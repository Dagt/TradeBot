from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import websockets

from .base import ExchangeAdapter
from .deribit import DeribitAdapter
from ..utils.metrics import WS_FAILURES, WS_RECONNECTS

log = logging.getLogger(__name__)


class DeribitWSAdapter(ExchangeAdapter):
    """Websocket wrapper delegando a :class:`DeribitAdapter` para REST."""

    name = "deribit_ws"

    def __init__(self, rest: DeribitAdapter | None = None, testnet: bool = False):
        super().__init__()
        self.rest = rest
        self.ws_url = (
            "wss://test.deribit.com/ws/api/v2" if testnet else "wss://www.deribit.com/ws/api/v2"
        )

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Stream trades from Deribit public websocket."""

        channel = f"trades.{symbol}"
        sub = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [channel]},
            "id": 1,
        }

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=20, ping_timeout=20
                ) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        params = msg.get("params") or {}
                        for t in params.get("data") or []:
                            price = float(t.get("price") or 0.0)
                            qty = float(t.get("amount") or t.get("size") or 0.0)
                            side = (t.get("direction") or t.get("side") or "").lower()
                            ts_ms = int(t.get("timestamp") or t.get("time") or 0)
                            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                            self.state.last_px[symbol] = price
                            yield self.normalize_trade(symbol, ts, price, qty, side)
            except Exception as e:
                WS_FAILURES.labels(adapter=self.name).inc()
                log.warning(
                    "Deribit trades WS desconectado (%s). Reintento en %.1fs ...",
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
                WS_RECONNECTS.labels(adapter=self.name).inc()
                backoff = min(backoff * 2, 30.0)

    async def stream_order_book(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Stream order book snapshots from Deribit."""

        channel = f"book.{depth}.{symbol}"
        sub = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [channel]},
            "id": 1,
        }

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=20, ping_timeout=20
                ) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = (msg.get("params") or {}).get("data") or {}
                        bids = [[float(p), float(q)] for p, q, *_ in data.get("bids", [])]
                        asks = [[float(p), float(q)] for p, q, *_ in data.get("asks", [])]
                        ts_ms = int(data.get("timestamp") or data.get("ts") or 0)
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                        self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                        yield self.normalize_order_book(symbol, ts, bids, asks)
            except Exception as e:
                WS_FAILURES.labels(adapter=self.name).inc()
                log.warning(
                    "Deribit book WS desconectado (%s). Reintento en %.1fs ...",
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
                WS_RECONNECTS.labels(adapter=self.name).inc()
                backoff = min(backoff * 2, 30.0)

    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_funding(symbol)
        raise NotImplementedError("Funding no disponible")

    async def fetch_basis(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_basis(symbol)
        raise NotImplementedError("Basis no disponible")

    async def fetch_oi(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_oi(symbol)
        raise NotImplementedError("Open interest no disponible")

    fetch_open_interest = fetch_oi

    async def place_order(self, *args, **kwargs) -> dict:
        if self.rest:
            return await self.rest.place_order(*args, **kwargs)
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str) -> dict:
        if self.rest:
            return await self.rest.cancel_order(order_id)
        raise NotImplementedError("no aplica en WS")
