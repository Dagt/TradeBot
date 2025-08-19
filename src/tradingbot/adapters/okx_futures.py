# src/tradingbot/adapters/okx_futures.py
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

try:
    import ccxt.async_support as ccxt
except Exception:  # pragma: no cover
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings
from ..core.symbols import normalize
from ..utils.secrets import validate_scopes
from ..execution.venue_adapter import translate_order_flags

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

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = self.ws_public_url
        sym = normalize(symbol)
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
        url = self.ws_public_url
        sym = normalize(symbol)
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

        OKX proporciona tanto el ``markPx`` como el ``indexPx`` en el ticker
        público.  Se consulta vía ``fetchTicker`` y se calcula la diferencia
        entre ambos valores.  Si la API o el adaptador REST no están
        disponibles se lanza ``NotImplementedError`` para dejar claro que el
        venue no soporta esta métrica.
        """

        sym = normalize(symbol)
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
        basis = mark_px - index_px
        return {"ts": ts_dt, "basis": basis}

    async def fetch_oi(self, symbol: str):
        """Fetch open interest for the given contract ``symbol``.

        OKX exposes the value through the ``/public/open-interest`` endpoint.
        The result is a list under ``data`` with the current open interest and
        timestamp in milliseconds.  We normalise this into
        ``{"ts": datetime, "oi": float}``.
        """

        sym = normalize(symbol)
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
        iceberg_qty: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        reduce_only: bool = False,
        params: dict | None = None,
    ) -> dict:
        params = params or {}
        extra = translate_order_flags(
            self.name,
            post_only=post_only,
            time_in_force=time_in_force,
            iceberg_qty=iceberg_qty,
            take_profit=take_profit,
            stop_loss=stop_loss,
            reduce_only=reduce_only,
        )
        params.update(extra)
        return await self._request(
            self.rest.create_order, symbol, type_, side, qty, price, params
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        return await self._request(self.rest.cancel_order, order_id, symbol)
