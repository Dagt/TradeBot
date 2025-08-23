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
    """Lightweight OKX futures websocket adapter."""

    name = "okx_futures_ws"

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
        self.name = "okx_futures_ws_testnet" if testnet else "okx_futures_ws"

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Return OKX formatted perpetual contract symbol."""

        sym = ExchangeAdapter.normalize_symbol(symbol)
        if not sym:
            return sym
        parts = sym.split("-")
        base_quote = parts[0]
        suffix = parts[1] if len(parts) > 1 else "SWAP"
        quotes = [
            "USDT",
            "USDC",
            "USD",
            "BTC",
            "ETH",
            "EUR",
            "GBP",
            "JPY",
        ]
        quote = next((q for q in quotes if base_quote.endswith(q)), base_quote[-4:])
        base = base_quote[: -len(quote)]
        return f"{base}-{quote}-{suffix}"

    # ------------------------------------------------------------------
    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Yield normalised trades for ``symbol``."""

        url = self.ws_public_url
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [f"trades:{sym}"]}
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

    DEPTH_TO_CHANNEL = {
        1: "bbo-tbt",
        5: "books5",
        50: "books50-l2-tbt",
        400: "books-l2-tbt",
    }

    async def stream_order_book(self, symbol: str, depth: int = 5) -> AsyncIterator[dict]:
        """Yield L2 order book updates for ``symbol``."""

        url = self.ws_public_url
        sym = self._normalize(symbol)
        channel = self.DEPTH_TO_CHANNEL.get(depth)
        if channel is None:
            raise ValueError(f"depth must be one of {sorted(self.DEPTH_TO_CHANNEL)}")
        sub = {"op": "subscribe", "args": [f"{channel}:{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                if channel == "bbo-tbt":
                    bid_px = float(d.get("bidPx", 0))
                    bid_qty = float(d.get("bidSz", 0))
                    ask_px = float(d.get("askPx", 0))
                    ask_qty = float(d.get("askSz", 0))
                    bids = [[bid_px, bid_qty]] if bid_px else []
                    asks = [[ask_px, ask_qty]] if ask_px else []
                else:
                    bids = [[float(p), float(q)] for p, q, *_ in d.get("bids", [])]
                    asks = [[float(p), float(q)] for p, q, *_ in d.get("asks", [])]
                ts_ms = int(d.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Stream best bid/ask levels for ``symbol`` using the bbo channel."""

        url = self.ws_public_url
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [f"bbo-tbt:{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                bid_px = float(d.get("bidPx", 0))
                bid_qty = float(d.get("bidSz", 0))
                ask_px = float(d.get("askPx", 0))
                ask_qty = float(d.get("askSz", 0))
                ts_ms = int(d.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                yield {
                    "symbol": symbol,
                    "ts": ts,
                    "bid_px": bid_px,
                    "bid_qty": bid_qty,
                    "ask_px": ask_px,
                    "ask_qty": ask_qty,
                }

    async def stream_book_delta(self, symbol: str, depth: int = 5) -> AsyncIterator[dict]:
        """Stream order book deltas relative to previous snapshots."""

        prev: dict | None = None
        async for ob in self.stream_order_book(symbol, depth):
            curr_bids = list(zip(ob.get("bid_px", []), ob.get("bid_qty", [])))
            curr_asks = list(zip(ob.get("ask_px", []), ob.get("ask_qty", [])))
            if prev is None:
                delta_bids = curr_bids
                delta_asks = curr_asks
            else:
                prev_bids = dict(zip(prev.get("bid_px", []), prev.get("bid_qty", [])))
                prev_asks = dict(zip(prev.get("ask_px", []), prev.get("ask_qty", [])))
                delta_bids = [[p, q] for p, q in curr_bids if prev_bids.get(p) != q]
                delta_bids += [[p, 0.0] for p in prev_bids.keys() - {p for p, _ in curr_bids}]
                delta_asks = [[p, q] for p, q in curr_asks if prev_asks.get(p) != q]
                delta_asks += [[p, 0.0] for p in prev_asks.keys() - {p for p, _ in curr_asks}]
            prev = ob
            yield {
                "symbol": symbol,
                "ts": ob.get("ts"),
                "bid_px": [p for p, _ in delta_bids],
                "bid_qty": [q for _, q in delta_bids],
                "ask_px": [p for p, _ in delta_asks],
                "ask_qty": [q for _, q in delta_asks],
            }

    async def stream_funding(self, symbol: str) -> AsyncIterator[dict]:
        """Stream funding rate updates for ``symbol``."""

        url = self.ws_public_url
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [f"funding-rate:{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                rate = d.get("fundingRate") or d.get("fundingRate")
                if rate is None:
                    continue
                ts_ms = int(d.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                interval = int(d.get("fundingInterval") or 0)
                yield {
                    "symbol": symbol,
                    "ts": ts,
                    "rate": float(rate),
                    "interval_sec": interval,
                }

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Stream open interest updates for ``symbol``."""

        url = self.ws_public_url
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [f"open-interest:{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                oi = d.get("oi") or d.get("openInterest")
                if oi is None:
                    continue
                ts_ms = int(d.get("ts", 0))
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                yield {"symbol": symbol, "ts": ts, "oi": float(oi)}

    # ------------------------------------------------------------------
    async def fetch_funding(self, symbol: str):
        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para funding")
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "publicGetPublicFundingRate", None)
        if method is not None:
            data = await self._request(method, {"instId": sym})
            lst = data.get("data") or []
            item = lst[0] if lst else {}
            ts_ms = int(
                item.get("fundingTime") or item.get("ts") or item.get("timestamp") or 0
            )
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rate = float(item.get("fundingRate") or item.get("rate") or 0.0)
            return {"ts": ts, "rate": rate}

        # Fallback to legacy CCXT ``fetchFundingRate`` returning a single dict
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:  # pragma: no cover
            raise NotImplementedError("Funding no soportado")
        legacy_sym = ExchangeAdapter.normalize_symbol(symbol)
        data = await self._request(method, legacy_sym)
        ts_ms = int(data.get("timestamp") or data.get("ts") or 0)
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rate = float(data.get("fundingRate") or data.get("rate") or 0.0)
        return {"ts": ts, "rate": rate}

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
        sym = ExchangeAdapter.normalize_symbol(symbol)
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
