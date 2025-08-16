# src/tradingbot/adapters/okx_futures.py
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import websockets

try:
    import ccxt
except Exception:  # pragma: no cover
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings
from ..utils.secrets import validate_scopes

log = logging.getLogger(__name__)


class OKXFuturesAdapter(ExchangeAdapter):
    """Adapter para futuros/swap perpetuos de OKX."""

    name = "okx_futures"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        testnet: bool = False,
    ):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")
        # "swap" cubre contratos perpetuos
        key = api_key or (settings.okx_testnet_api_key if testnet else settings.okx_api_key)
        secret = api_secret or (settings.okx_testnet_api_secret if testnet else settings.okx_api_secret)
        passphrase = api_passphrase or (
            settings.okx_testnet_api_passphrase if testnet else settings.okx_api_passphrase
        )

        self.rest = ccxt.okx({
            "apiKey": key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.rest.set_sandbox_mode(testnet)
        # Validar permisos disponibles
        validate_scopes(self.rest, log)

        self.ws_public_url = (
            "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
            if testnet
            else "wss://ws.okx.com:8443/ws/v5/public"
        )
        self.name = "okx_futures_testnet" if testnet else "okx_futures"

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [{"channel": "trades", "instId": sym}]}
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        for t in msg.get("data", []) or []:
                            price = float(t.get("px"))
                            qty = float(t.get("sz", 0))
                            side = t.get("side", "")
                            ts = datetime.fromtimestamp(int(t.get("ts", 0)) / 1000, tz=timezone.utc)
                            self.state.last_px[symbol] = price
                            yield self.normalize_trade(symbol, ts, price, qty, side)
            except Exception as e:  # pragma: no cover
                log.warning("OKX futures trade WS error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        url = self.ws_public_url
        sym = self.normalize_symbol(symbol)
        sub = {"op": "subscribe", "args": [{"channel": "books5", "instId": sym}]}
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        for d in msg.get("data", []) or []:
                            bids = [[float(p), float(q)] for p, q, *_ in d.get("bids", [])]
                            asks = [[float(p), float(q)] for p, q, *_ in d.get("asks", [])]
                            ts = datetime.fromtimestamp(int(d.get("ts", 0)) / 1000, tz=timezone.utc)
                            self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                            yield self.normalize_order_book(symbol, ts, bids, asks)
            except Exception as e:  # pragma: no cover
                log.warning("OKX futures book WS error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

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

    async def fetch_oi(self, symbol: str):
        """Fetch open interest for the given contract ``symbol``.

        OKX exposes the value through the ``/public/open-interest`` endpoint.
        The result is a list under ``data`` with the current open interest and
        timestamp in milliseconds.  We normalise this into
        ``{"ts": datetime, "oi": float}``.
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
        iceberg_qty: float | None = None,
    ) -> dict:
        params = {}
        if iceberg_qty is not None:
            params["iceberg"] = iceberg_qty
        return await self._to_thread(
            self.rest.create_order, symbol, type_, side, qty, price, params
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        return await self._to_thread(self.rest.cancel_order, order_id, symbol)
