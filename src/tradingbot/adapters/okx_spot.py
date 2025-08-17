# src/tradingbot/adapters/okx_spot.py
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

try:  # pragma: no cover - optional during tests
    import ccxt
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

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sym = self.normalize_symbol(symbol)
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

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [{"channel": "books5", "instId": sym}]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                bids = [[float(p), float(q)] for p, q, *_ in d.get("bids", [])]
                asks = [[float(p), float(q)] for p, q, *_ in d.get("asks", [])]
                ts = datetime.fromtimestamp(int(d.get("ts", 0)) / 1000, tz=timezone.utc)
                self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        sym = self.normalize_symbol(symbol)
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
        """Spot en OKX no ofrece una métrica de *basis*.

        La función existe únicamente para cumplir con la interfaz y dejar
        explícito que este dato no está disponible en el venue.
        """

        raise NotImplementedError("Basis not supported in spot markets")

    async def fetch_oi(self, symbol: str):
        """Fetch open interest for the corresponding instrument.

        Similar to futures, OKX exposes spot open interest through the
        ``/public/open-interest`` endpoint.  We parse the first element in the
        ``data`` array and return it as ``{"ts": datetime, "oi": float}``.
        """

        sym = self.normalize_symbol(symbol)
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
        params: dict | None = None,
    ) -> dict:
        params = params or {}
        if post_only:
            params["postOnly"] = True
        if time_in_force:
            params["timeInForce"] = time_in_force
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
