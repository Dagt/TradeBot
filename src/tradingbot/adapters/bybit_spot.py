# src/tradingbot/adapters/bybit_spot.py
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

try:
    import ccxt.async_support as ccxt
except Exception:  # pragma: no cover - ccxt optional during tests
    ccxt = None

from .base import ExchangeAdapter
from ..core.symbols import normalize
from ..utils.secrets import validate_scopes

log = logging.getLogger(__name__)


class BybitSpotAdapter(ExchangeAdapter):
    """Adapter para Spot de Bybit usando WS público y REST via CCXT."""

    name = "bybit_spot"

    def __init__(self):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")
        # enableRateLimit respeta límites de la API
        self.rest = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        # Advertir si la clave carece de permisos de trade o tiene retiros habilitados
        validate_scopes(self.rest, log)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://stream.bybit.com/v5/public/spot"
        sym = normalize(symbol)
        sub = {"op": "subscribe", "args": [f"publicTrade.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for t in msg.get("data", []) or []:
                price = float(t.get("p"))
                qty = float(t.get("v", 0))
                side = t.get("S", "").lower()
                ts = datetime.fromtimestamp(int(t.get("T", 0)) / 1000, tz=timezone.utc)
                self.state.last_px[symbol] = price
                yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://stream.bybit.com/v5/public/spot"
        sym = normalize(symbol)
        sub = {"op": "subscribe", "args": [f"orderbook.1.{sym}"]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            data = msg.get("data") or {}
            if not data:
                continue
            bids = [[float(p), float(q)] for p, q, *_ in data.get("b", [])]
            asks = [[float(p), float(q)] for p, q, *_ in data.get("a", [])]
            ts = datetime.fromtimestamp(int(data.get("ts", 0)) / 1000, tz=timezone.utc)
            self.state.order_book[symbol] = {"bids": bids, "asks": asks}
            yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        sym = normalize(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:
            raise NotImplementedError("Funding not supported")
        data = await self._request(method, sym)
        ts = data.get("timestamp") or data.get("time") or data.get("ts") or 0
        ts = int(ts)
        if ts > 1e12:
            ts /= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        rate = float(data.get("fundingRate") or data.get("rate") or data.get("value") or 0.0)
        return {"ts": ts_dt, "rate": rate}

    async def fetch_basis(self, symbol: str):
        """Return the basis (mark - index) for ``symbol``.

        Even though spot markets do not trade a perpetual contract, Bybit
        exposes the mark and index price of the corresponding linear contract
        via ``public/v5/market/premium-index-price``.  We reuse that endpoint to
        compute the difference and return it normalised as
        ``{"ts": datetime, "basis": float}``.
        """

        sym = normalize(symbol)
        method = getattr(self.rest, "publicGetV5MarketPremiumIndexPrice", None)
        if method is None:
            raise NotImplementedError("Basis not supported")

        data = await self._request(method, {"category": "linear", "symbol": sym})
        lst = (data.get("result") or {}).get("list") or []
        item = lst[0] if lst else {}
        ts_ms = int(item.get("timestamp") or item.get("ts") or data.get("time", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        mark_px = float(
            item.get("markPrice")
            or item.get("mark_price")
            or data.get("markPrice")
            or 0.0
        )
        index_px = float(
            item.get("indexPrice")
            or item.get("index_price")
            or data.get("indexPrice")
            or 0.0
        )
        basis = mark_px - index_px
        return {"ts": ts, "basis": basis}

    async def fetch_oi(self, symbol: str):
        """Return current open interest for ``symbol``.

        Spot markets do not have an intrinsic open interest, but Bybit's public
        API exposes it for the corresponding linear contract.  We call the same
        ``public/v5/market/open-interest`` endpoint used for futures and return
        the data normalised to ``{"ts": datetime, "oi": float}``.
        """

        sym = normalize(symbol)
        method = getattr(self.rest, "publicGetV5MarketOpenInterest", None)
        if method is None:
            raise NotImplementedError("Open interest not supported")

        data = await self._request(method, {"category": "linear", "symbol": sym})
        lst = (data.get("result") or {}).get("list") or []
        item = lst[0] if lst else {}
        ts_ms = int(item.get("timestamp", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(item.get("openInterest", 0.0))
        return {"ts": ts, "oi": oi}

    async def place_order(self, symbol: str, side: str, type_: str, qty: float,
                          price: float | None = None) -> dict:
        return await self.rest.create_order(symbol, type_, side, qty, price)

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        return await self.rest.cancel_order(order_id, symbol)
