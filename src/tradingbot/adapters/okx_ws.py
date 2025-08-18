# src/tradingbot/adapters/okx_ws.py
"""Websocket adapter for OKX perpetual swaps.

The implementation mirrors :mod:`binance_spot_ws` but targets OKX's v5
public websocket API.  It provides trade and order book streams and delegates
funding, basis and open interest queries to an optional REST adapter.  The
underlying websocket connection reuses :func:`ExchangeAdapter._ws_messages`
which already implements ping/pong and exponential backoff reconnects."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from .base import ExchangeAdapter

log = logging.getLogger(__name__)


class OKXWSAdapter(ExchangeAdapter):
    """Lightweight OKX websocket adapter."""

    name = "okx_ws"

    def __init__(self, ws_base: str | None = None, rest: ExchangeAdapter | None = None, testnet: bool = False):
        super().__init__()
        if ws_base:
            self.ws_public_url = ws_base
        else:
            self.ws_public_url = (
                "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
                if testnet
                else "wss://ws.okx.com:8443/ws/v5/public"
            )
        self.rest = rest
        self.name = "okx_ws_testnet" if testnet else "okx_ws"

    # ------------------------------------------------------------------
    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Yield normalised trades for ``symbol``."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [{"channel": "trades", "instId": sym}]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for t in msg.get("data", []) or []:
                price_raw = t.get("px")
                if price_raw is None:
                    continue
                price = float(price_raw)
                qty = float(t.get("sz", 0))
                side = t.get("side", "")
                ts_ms = int(t.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                self.state.last_px[symbol] = price
                yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_order_book(self, symbol: str, depth: int = 5) -> AsyncIterator[dict]:
        """Yield L2 order book updates for ``symbol``."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        channel = f"books{depth}" if depth in (1, 5, 10, 25) else "books5"
        sub = {"op": "subscribe", "args": [{"channel": channel, "instId": sym}]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                bids = [[float(p), float(q)] for p, q, *_ in d.get("bids", [])]
                asks = [[float(p), float(q)] for p, q, *_ in d.get("asks", [])]
                ts_ms = int(d.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    # ------------------------------------------------------------------
    async def fetch_funding(self, symbol: str):
        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para funding")
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:  # pragma: no cover
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
        method = getattr(self.rest, "fetchTicker", None)
        if method is None:  # pragma: no cover
            raise NotImplementedError("Basis no soportado")
        data = await self._request(method, sym)
        ts = int(data.get("timestamp") or data.get("ts") or data.get("time") or 0)
        if ts > 1e12:
            ts //= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        mark_px = float(
            data.get("markPrice")
            or data.get("markPx")
            or data.get("mark_price")
            or data.get("last")
            or 0.0
        )
        index_px = float(
            data.get("indexPrice")
            or data.get("indexPx")
            or data.get("index_price")
            or 0.0
        )
        return {"ts": ts_dt, "basis": mark_px - index_px}

    async def fetch_oi(self, symbol: str):
        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para open interest")
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "publicGetPublicOpenInterest", None)
        if method is None:  # pragma: no cover
            raise NotImplementedError("Open interest no soportado")
        data = await self._request(method, {"instId": sym})
        lst = data.get("data") or []
        item = lst[0] if lst else {}
        ts_ms = int(item.get("ts", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(item.get("oi", 0.0))
        return {"ts": ts, "oi": oi}

    # ------------------------------------------------------------------
    async def place_order(self, *args, **kwargs) -> dict:
        if self.rest:
            # Use `_request` so both sync and async REST clients are supported
            return await self._request(self.rest.place_order, *args, **kwargs)
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str, *args, **kwargs) -> dict:
        if self.rest:
            return await self._request(self.rest.cancel_order, order_id, *args, **kwargs)
        raise NotImplementedError("no aplica en WS")
