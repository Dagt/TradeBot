from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

try:  # pragma: no cover - optional dependency during some tests
    import ccxt.async_support as ccxt
except Exception:  # pragma: no cover
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings
from ..core.symbols import normalize
from ..utils.secrets import validate_scopes

log = logging.getLogger(__name__)


class DeribitAdapter(ExchangeAdapter):
    """Adapter simple para Deribit (perpetuos)."""

    name = "deribit"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = False,
    ):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")

        key = api_key or getattr(settings, "deribit_testnet_api_key" if testnet else "deribit_api_key", None)
        secret = api_secret or getattr(settings, "deribit_testnet_api_secret" if testnet else "deribit_api_secret", None)

        self.rest = ccxt.deribit({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
        })
        self.rest.set_sandbox_mode(testnet)
        validate_scopes(self.rest, log)
        self.name = "deribit_testnet" if testnet else "deribit"

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        while True:  # poll trades vía REST; Deribit no posee WS en este adaptador
            data = await self._request(self.rest.fetch_trades, symbol, limit=1)
            for t in data:
                price = float(t.get("price", 0.0))
                qty = float(t.get("amount") or t.get("size") or 0.0)
                ts_ms = int(t.get("timestamp") or 0)
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                side = t.get("side") or t.get("direction")
                self.state.last_px[symbol] = price
                yield self.normalize_trade(symbol, ts, price, qty, side)
            await asyncio.sleep(1)

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:  # pragma: no cover - not supported
        raise NotImplementedError("BBA stream not supported")

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:  # pragma: no cover - not supported
        raise NotImplementedError("Order book delta stream not supported")

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
        sym = normalize(symbol)
        method = (
            getattr(self.rest, "public_get_get_funding_rate", None)
            or getattr(self.rest, "fetchFundingRate", None)
        )
        if method is None:
            raise NotImplementedError("Funding not supported")
        params = {"instrument_name": sym} if "public_get" in method.__name__.lower() else sym
        data = await self._request(method, params)
        res = data.get("result") or data
        ts = int(res.get("timestamp") or res.get("time") or 0)
        ts_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        rate = float(res.get("funding_rate") or res.get("fundingRate") or res.get("rate") or 0.0)
        return {"ts": ts_dt, "rate": rate}

    async def fetch_basis(self, symbol: str):
        sym = normalize(symbol)
        method = getattr(self.rest, "public_get_ticker", None) or getattr(self.rest, "fetchTicker", None)
        if method is None:
            raise NotImplementedError("Basis not supported")
        params = {"instrument_name": sym} if "public_get" in method.__name__.lower() else sym
        data = await self._request(method, params)
        res = data.get("result") or data
        ts = int(res.get("timestamp") or res.get("time") or 0)
        ts_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        index_px = float(res.get("index_price") or res.get("underlying_price") or 0.0)
        mark_px = float(res.get("mark_price") or res.get("last_price") or res.get("price") or 0.0)
        basis = mark_px - index_px
        return {"ts": ts_dt, "basis": basis}

    async def fetch_oi(self, symbol: str):
        sym = normalize(symbol)
        method = (
            getattr(self.rest, "public_get_get_open_interest", None)
            or getattr(self.rest, "fetchOpenInterest", None)
        )
        if method is None:
            raise NotImplementedError("Open interest not supported")
        params = {"instrument_name": sym} if "public_get" in method.__name__.lower() else sym
        data = await self._request(method, params)
        res = data.get("result") or data
        ts = int(res.get("timestamp") or res.get("time") or 0)
        ts_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        oi = float(res.get("open_interest") or res.get("openInterest") or res.get("oi") or 0.0)
        return {"ts": ts_dt, "oi": oi}

    fetch_open_interest = fetch_oi

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
    ) -> dict:
        params = {}
        if post_only:
            params["post_only"] = True
        if time_in_force:
            params["time_in_force"] = time_in_force
        if reduce_only:
            params["reduce_only"] = True
        return await self._request(self.rest.create_order, symbol, type_, side, qty, price, params)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._request(self.rest.cancel_order, order_id)
