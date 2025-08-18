# src/tradingbot/adapters/bybit_ws.py
"""Lightweight websocket adapter for Bybit perpetual futures.

Based on the `binance_spot_ws` implementation, this adapter focuses on
streaming trades and L2 order book snapshots while delegating optional REST
queries for funding, basis and open interest to an injected REST adapter.

Only the public websocket is used; ping/pong handling and reconnect backoff
are provided by :func:`ExchangeAdapter._ws_messages` from the base class.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from .base import ExchangeAdapter

log = logging.getLogger(__name__)


class BybitWSAdapter(ExchangeAdapter):
    """Websocket adapter for Bybit USDT perpetuals.

    The adapter exposes a small subset of functionality required by the
    project: trade and order book streams plus helper methods for funding,
    basis and open interest retrieval via an optional REST client.
    """

    name = "bybit_ws"

    def __init__(self, ws_base: str | None = None, rest: ExchangeAdapter | None = None, testnet: bool = False):
        super().__init__()
        if ws_base:
            self.ws_public_url = ws_base
        else:
            self.ws_public_url = (
                "wss://stream-testnet.bybit.com/v5/public/linear"
                if testnet
                else "wss://stream.bybit.com/v5/public/linear"
            )
        self.rest = rest
        self.name = "bybit_ws_testnet" if testnet else "bybit_ws"

    # ------------------------------------------------------------------
    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Yield normalised trade dictionaries for ``symbol``."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"publicTrade.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for t in msg.get("data", []) or []:
                price_raw = t.get("p")
                if price_raw is None:
                    continue
                price = float(price_raw)
                qty = float(t.get("v", 0))
                side = (t.get("S", "")).lower()
                ts_ms = int(t.get("T", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                self.state.last_px[symbol] = price
                yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_order_book(self, symbol: str, depth: int = 1) -> AsyncIterator[dict]:
        """Yield L2 order book snapshots for ``symbol``."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"orderbook.{depth}.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            data = msg.get("data") or {}
            if not data:
                continue
            bids = [[float(p), float(q)] for p, q, *_ in data.get("b", [])]
            asks = [[float(p), float(q)] for p, q, *_ in data.get("a", [])]
            ts_ms = int(data.get("ts", 0))
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            self.state.order_book[symbol] = {"bids": bids, "asks": asks}
            yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    # ------------------------------------------------------------------
    async def fetch_funding(self, symbol: str):
        """Return current funding rate for ``symbol`` via REST."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para funding")
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:  # pragma: no cover - depende del REST
            raise NotImplementedError("Funding no soportado")
        data = await self._request(method, sym)
        ts = int(data.get("timestamp") or data.get("time") or data.get("ts") or 0)
        if ts > 1e12:
            ts //= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        rate = float(data.get("fundingRate") or data.get("rate") or data.get("value") or 0.0)
        return {"ts": ts_dt, "rate": rate}

    async def fetch_basis(self, symbol: str):
        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para basis")
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "publicGetV5MarketPremiumIndexPrice", None)
        if method is None:  # pragma: no cover
            raise NotImplementedError("Basis no soportado")
        data = await self._request(method, {"category": "linear", "symbol": sym})
        lst = (data.get("result") or {}).get("list") or []
        item = lst[0] if lst else {}
        ts_ms = int(item.get("timestamp") or item.get("ts") or data.get("time", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        mark_px = float(item.get("markPrice") or data.get("markPrice") or 0.0)
        index_px = float(item.get("indexPrice") or data.get("indexPrice") or 0.0)
        return {"ts": ts, "basis": mark_px - index_px}

    async def fetch_oi(self, symbol: str):
        """Return current open interest for ``symbol``."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para open interest")
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "publicGetV5MarketOpenInterest", None)
        if method is None:  # pragma: no cover
            raise NotImplementedError("Open interest no soportado")
        data = await self._request(method, {"category": "linear", "symbol": sym})
        lst = (data.get("result") or {}).get("list") or []
        item = lst[0] if lst else {}
        ts_ms = int(item.get("timestamp", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(item.get("openInterest", 0.0))
        return {"ts": ts, "oi": oi}

    # ------------------------------------------------------------------
    async def place_order(self, *args, **kwargs) -> dict:
        if self.rest:
            # Delegate through `_request` to support both sync and async
            # REST implementations (``DummyRest`` in tests is synchronous).
            return await self._request(self.rest.place_order, *args, **kwargs)
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str, *args, **kwargs) -> dict:
        if self.rest:
            return await self._request(self.rest.cancel_order, order_id, *args, **kwargs)
        raise NotImplementedError("no aplica en WS")
