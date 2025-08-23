# src/tradingbot/adapters/bybit_ws.py
"""WebSocket adapter for Bybit USDT perpetual futures.

Streams trades and L2 order book snapshots and can leverage a REST client for
auxiliary data such as funding and open interest.
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

    name = "bybit_futures_ws"
    # Bybit's public websocket does not expose funding or open interest
    # channels (the server responds with ``error:handler not found``).  Only
    # native streams are advertised here; users should rely on the REST
    # adapter for those additional kinds.
    supported_kinds = {"trades", "orderbook", "bba", "delta"}

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
        self.name = "bybit_futures_ws_testnet" if testnet else "bybit_futures_ws"

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
        """Yield normalised L2 order book snapshots for ``symbol``.

        Bybit's ``orderbook.{depth}`` channel first sends a full snapshot and
        subsequently publishes incremental updates.  We apply those deltas
        locally to maintain the book and emit the reconstructed snapshot each
        time an update is received.
        """

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"orderbook.{depth}.{sym}"]}
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}

        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            data = msg.get("data") or {}
            if not data:
                continue

            side_updates = {
                "b": bids,
                "a": asks,
            }

            if msg.get("type") == "snapshot":
                for side_key, book in side_updates.items():
                    updates = data.get(side_key, [])
                    book.clear()
                    for p, q, *_ in updates:
                        book[float(p)] = float(q)
            elif msg.get("type") == "delta":
                for side_key, book in side_updates.items():
                    for p, q, *_ in data.get(side_key, []):
                        price = float(p)
                        qty = float(q)
                        if qty == 0:
                            book.pop(price, None)
                        else:
                            book[price] = qty
            else:
                # subscription acknowledgements or unknown messages
                continue

            ts_ms = int(data.get("ts", 0))
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

            bids_sorted = sorted(bids.items(), key=lambda x: -x[0])[:depth]
            asks_sorted = sorted(asks.items(), key=lambda x: x[0])[:depth]
            bids_list = [[p, q] for p, q in bids_sorted]
            asks_list = [[p, q] for p, q in asks_sorted]

            self.state.order_book[symbol] = {"bids": bids_list, "asks": asks_list}
            yield self.normalize_order_book(symbol, ts, bids_list, asks_list)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Stream best bid/ask updates for ``symbol`` using the ticker feed."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"tickers.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                bid_px = d.get("bid1Price")
                ask_px = d.get("ask1Price")
                if bid_px is None and ask_px is None:
                    continue
                ts_ms = int(d.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                yield {
                    "symbol": symbol,
                    "ts": ts,
                    "bid_px": float(bid_px) if bid_px is not None else None,
                    "bid_qty": float(d.get("bid1Size", 0.0)),
                    "ask_px": float(ask_px) if ask_px is not None else None,
                    "ask_qty": float(d.get("ask1Size", 0.0)),
                }

    async def stream_book_delta(self, symbol: str, depth: int = 1) -> AsyncIterator[dict]:
        """Stream order book deltas for ``symbol``.

        The ``orderbook`` channel itself provides incremental updates.  We
        subscribe separately and emit the changed levels directly without
        reconstructing the full book.
        """

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
            yield {
                "symbol": symbol,
                "ts": ts,
                "bid_px": [p for p, _ in bids],
                "bid_qty": [q for _, q in bids],
                "ask_px": [p for p, _ in asks],
                "ask_qty": [q for _, q in asks],
            }

    async def stream_funding(self, symbol: str) -> AsyncIterator[dict]:
        """Stream funding rate updates for ``symbol``."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"fundingRate.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            if msg.get("op") == "subscribe" and not msg.get("success", True):
                raise NotImplementedError("fundingRate stream not supported")
            for d in msg.get("data", []) or []:
                rate = d.get("fundingRate") or d.get("rate")
                if rate is None:
                    continue
                ts_ms = int(d.get("timestamp") or d.get("ts") or 0)
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                interval = int(d.get("fundingRateInterval") or 0)
                yield {
                    "symbol": symbol,
                    "ts": ts,
                    "rate": float(rate),
                    "interval_sec": interval,
                }

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Stream open interest updates for ``symbol``."""

        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [f"openInterest.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            if msg.get("op") == "subscribe" and not msg.get("success", True):
                raise NotImplementedError("openInterest stream not supported")
            for d in msg.get("data", []) or []:
                oi = d.get("openInterest") or d.get("oi")
                if oi is None:
                    continue
                ts_ms = int(d.get("timestamp") or d.get("ts") or 0)
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                yield {"symbol": symbol, "ts": ts, "oi": float(oi)}

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
