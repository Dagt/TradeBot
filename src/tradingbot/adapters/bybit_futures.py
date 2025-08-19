# src/tradingbot/adapters/bybit_futures.py
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
except Exception:  # pragma: no cover - ccxt optional durante tests
    ccxt = None
    NetworkError = ExchangeError = Exception

from .base import ExchangeAdapter
from ..config import settings
from ..core.symbols import normalize
from ..utils.secrets import validate_scopes
from ..execution.venue_adapter import translate_order_flags

log = logging.getLogger(__name__)


class BybitFuturesAdapter(ExchangeAdapter):
    """Adapter para futuros/perpetuos USDT de Bybit."""

    name = "bybit_futures"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = False,
    ):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")

        key = api_key or (settings.bybit_testnet_api_key if testnet else settings.bybit_api_key)
        secret = api_secret or (settings.bybit_testnet_api_secret if testnet else settings.bybit_api_secret)

        self.rest = ccxt.bybit({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.rest.set_sandbox_mode(testnet)
        # Validar permisos de la clave
        validate_scopes(self.rest, log)

        self.ws_public_url = (
            "wss://stream-testnet.bybit.com/v5/public/linear"
            if testnet
            else "wss://stream.bybit.com/v5/public/linear"
        )
        self.name = "bybit_futures_testnet" if testnet else "bybit_futures"

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = self.ws_public_url
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
        url = self.ws_public_url
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

    # Backwards compatible alias
    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        sym = normalize(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:  # pragma: no cover - depends on ccxt support
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

        Bybit exposes both the mark and index prices via the
        ``v5/market/premium-index-price`` endpoint.  The response nests the
        data under ``result.list``.  We fetch both prices and return their
        difference along with the corresponding timestamp.  If the REST
        adapter does not implement this endpoint, ``NotImplementedError`` is
        raised to signal that the venue does not support basis retrieval.
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

        Bybit provides the value under the ``public/v5/market/open-interest``
        endpoint.  The response wraps the data inside ``result.list``.  We
        normalise it into a lightweight ``{"ts": datetime, "oi": float}``
        structure.
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
                log.warning("Bybit place_order error %s, retrying in %.1fs", e, backoff)
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
                log.warning("Bybit cancel_order error %s, retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except ExchangeError:
                raise
