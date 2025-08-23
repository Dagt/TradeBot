# src/tradingbot/adapters/okx_spot.py
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

try:  # pragma: no cover - optional during tests
    import ccxt.async_support as ccxt
    NetworkError = getattr(ccxt, "NetworkError", Exception)
    ExchangeError = getattr(ccxt, "ExchangeError", Exception)
except Exception:  # pragma: no cover
    ccxt = None
    NetworkError = ExchangeError = Exception

from .base import ExchangeAdapter
from ..utils.secrets import validate_scopes

log = logging.getLogger(__name__)


class OKXSpotAdapter(ExchangeAdapter):
    """Adapter para Spot de OKX usando websockets y CCXT."""

    name = "okx_spot"

    def __init__(self):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")
        self.rest = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        # Validar permisos disponibles en la API key
        validate_scopes(self.rest, log)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Return OKX formatted spot symbol."""

        sym = ExchangeAdapter.normalize_symbol(symbol)
        if not sym:
            return sym
        parts = sym.split("-")
        base_quote = parts[0]
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
        return f"{base}-{quote}"

    def _normalize(self, symbol: str) -> str:
        """Return normalised symbol in OKX format."""

        return self.normalize_symbol(symbol).replace("/", "-").upper()

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [{"channel": "trades", "instId": sym}]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for t in msg.get("data", []) or []:
                price = float(t.get("px"))
                qty = float(t.get("sz", 0))
                side = t.get("side", "")
                ts = datetime.fromtimestamp(int(t.get("ts", 0)) / 1000, tz=timezone.utc)
                self.state.last_px[symbol] = price
                yield self.normalize_trade(symbol, ts, price, qty, side)

    DEPTH_TO_CHANNEL = {
        1: "bbo-tbt",
        5: "books5",
        50: "books50-l2-tbt",
        400: "books-l2-tbt",
    }

    async def stream_order_book(self, symbol: str, depth: int = 5) -> AsyncIterator[dict]:
        url = "wss://ws.okx.com:8443/ws/v5/public"
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
                ts = datetime.fromtimestamp(int(d.get("ts", 0)) / 1000, tz=timezone.utc)
                self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Emit best bid/ask quotes for ``symbol`` using bbo channel."""

        url = "wss://ws.okx.com:8443/ws/v5/public"
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [f"bbo-tbt:{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                bid = float(d.get("bidPx", 0))
                ask = float(d.get("askPx", 0))
                ts = datetime.fromtimestamp(int(d.get("ts", 0)) / 1000, tz=timezone.utc)
                yield {"symbol": symbol, "ts": ts, "bid": bid, "ask": ask}

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Yield incremental order book updates for ``symbol``."""

        prev: dict | None = None
        async for ob in self.stream_order_book(symbol, depth):
            curr_bids = list(zip(ob.get("bid_px", []), ob.get("bid_qty", [])))
            curr_asks = list(zip(ob.get("ask_px", []), ob.get("ask_qty", [])))
            if prev is None:
                delta_bids = curr_bids
                delta_asks = curr_asks
            else:
                pb = dict(zip(prev.get("bid_px", []), prev.get("bid_qty", [])))
                pa = dict(zip(prev.get("ask_px", []), prev.get("ask_qty", [])))
                delta_bids = [[p, q] for p, q in curr_bids if pb.get(p) != q]
                delta_bids += [[p, 0.0] for p in pb.keys() - {p for p, _ in curr_bids}]
                delta_asks = [[p, q] for p, q in curr_asks if pa.get(p) != q]
                delta_asks += [[p, 0.0] for p in pa.keys() - {p for p, _ in curr_asks}]
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
        """Poll funding rate updates via REST."""

        while True:
            data = await self.fetch_funding(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Poll open interest updates via REST."""

        while True:
            data = await self.fetch_oi(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def fetch_funding(self, symbol: str):
        sym = self._normalize(symbol)
        method = getattr(self.rest, "publicGetPublicFundingRate", None)
        if method is None:
            raise NotImplementedError("Funding not supported")
        data = await self._request(method, {"instId": sym})
        lst = data.get("data") or []
        item = lst[0] if lst else {}
        ts_ms = int(
            item.get("fundingTime") or item.get("ts") or item.get("timestamp") or 0
        )
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rate = float(item.get("fundingRate") or item.get("rate") or 0.0)
        return {"ts": ts, "rate": rate}

    async def fetch_basis(self, symbol: str):
        """Return the basis (mark - index) for ``symbol``.

        OKX's spot ticker exposes both the last traded price and an index
        price.  We call :meth:`fetchTicker` via CCXT and calculate the
        difference, returning a normalised ``{"ts": datetime, "basis": float}``.
        """

        sym = self._normalize(symbol)
        method = getattr(self.rest, "fetchTicker", None)
        if method is None:
            raise NotImplementedError("Basis not supported")

        data = await self._request(method, sym)
        ts = data.get("timestamp") or data.get("ts") or data.get("time") or 0
        ts = int(ts)
        if ts > 1e12:
            ts //= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        mark_px = float(
            data.get("markPrice")
            or data.get("last")
            or data.get("close")
            or 0.0
        )
        index_px = float(
            data.get("indexPrice")
            or data.get("index")
            or (data.get("info") or {}).get("idxPx", 0.0)
        )
        basis = mark_px - index_px
        return {"ts": ts_dt, "basis": basis}

    async def fetch_oi(self, symbol: str):
        """Fetch open interest for the corresponding instrument.

        Similar to futures, OKX exposes spot open interest through the
        ``/public/open-interest`` endpoint.  We parse the first element in the
        ``data`` array and return it as ``{"ts": datetime, "oi": float}``.
        """

        sym = self._normalize(symbol)
        method = getattr(self.rest, "publicGetPublicOpenInterest", None)
        if method is None:
            raise NotImplementedError("Open interest not supported")

        data = await self._request(method, {"instId": sym})
        lst = data.get("data") or []
        item = lst[0] if lst else {}
        ts_ms = int(item.get("ts", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(item.get("oi", 0.0))
        return {"ts": ts, "oi": oi}

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
        params: dict | None = None,
    ) -> dict:
        params = params or {}
        if post_only:
            params["postOnly"] = True
        if time_in_force:
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = True
        backoff = 1.0
        while True:
            try:
                return await self._request(
                    self.rest.create_order,
                    symbol,
                    type_,
                    side,
                    qty,
                    price,
                    params,
                )
            except NetworkError as e:  # pragma: no cover - network paths
                log.warning("OKX place_order error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except ExchangeError:
                raise

    async def cancel_order(
        self,
        order_id: str,
        symbol: str | None = None,
        params: dict | None = None,
    ) -> dict:
        params = params or {}
        backoff = 1.0
        while True:
            try:
                return await self._request(
                    self.rest.cancel_order,
                    order_id,
                    symbol,
                    params,
                )
            except NetworkError as e:  # pragma: no cover - network paths
                log.warning("OKX cancel_order error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except ExchangeError:
                raise
