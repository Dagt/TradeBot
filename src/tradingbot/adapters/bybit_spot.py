# src/tradingbot/adapters/bybit_spot.py
from __future__ import annotations

import asyncio
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
from ..config import settings

log = logging.getLogger(__name__)


class BybitSpotAdapter(ExchangeAdapter):
    """Adapter para Spot de Bybit usando WS público y REST via CCXT."""

    name = "bybit_spot"

    def __init__(
        self,
        testnet: bool = False,
        maker_fee_bps: float | None = None,
        taker_fee_bps: float | None = None,
    ):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")
        key = settings.bybit_testnet_api_key if testnet else settings.bybit_api_key
        secret = settings.bybit_testnet_api_secret if testnet else settings.bybit_api_secret
        self.rest = ccxt.bybit(
            {
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
                "testnet": testnet,
            }
        )
        self.rest.options["defaultType"] = "spot"
        self.rest.set_sandbox_mode(testnet)
        # Advertir si la clave carece de permisos de trade o tiene retiros habilitados
        validate_scopes(self.rest, log)
        self.ws_public_url = (
            "wss://stream-testnet.bybit.com/v5/public/spot"
            if testnet
            else "wss://stream.bybit.com/v5/public/spot"
        )
        self.name = "bybit_spot_testnet" if testnet else "bybit_spot"
        self.maker_fee_bps = float(
            maker_fee_bps
            if maker_fee_bps is not None
            else (
                settings.bybit_spot_testnet_maker_fee_bps if testnet else settings.bybit_spot_maker_fee_bps
            )
        )
        self.taker_fee_bps = float(
            taker_fee_bps
            if taker_fee_bps is not None
            else (
                settings.bybit_spot_testnet_taker_fee_bps if testnet else settings.bybit_spot_taker_fee_bps
            )
        )

        self._start_fee_updater()

    async def update_fees(self, symbol: str | None = None) -> None:
        params = {"category": "spot"}
        if symbol:
            params["symbol"] = normalize(symbol)
        try:
            res = await self._request(self.rest.privateGetV5AccountFeeRate, params)
            data = (res.get("result", {}).get("list") or [])[0]
            self.maker_fee_bps = abs(float(data.get("makerFeeRate", 0.0))) * 10000
            self.taker_fee_bps = abs(float(data.get("takerFeeRate", 0.0))) * 10000
        except Exception as e:  # pragma: no cover
            log.warning("update_fees failed: %s", e)

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
            ts_ms = int(msg.get("ts") or data.get("ts", 0))
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            self.state.order_book[symbol] = {"bids": bids, "asks": asks}
            yield self.normalize_order_book(symbol, ts, bids, asks)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Emit best bid/ask quotes for ``symbol``."""

        async for ob in self.stream_order_book(symbol):
            bid = ob.get("bid_px", [None])[0]
            bid_qty = ob.get("bid_qty", [None])[0]
            ask = ob.get("ask_px", [None])[0]
            ask_qty = ob.get("ask_qty", [None])[0]
            yield {
                "symbol": symbol,
                "ts": ob.get("ts"),
                "bid_px": bid,
                "bid_qty": bid_qty,
                "ask_px": ask,
                "ask_qty": ask_qty,
            }

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Yield incremental book updates for ``symbol``."""

        prev: dict | None = None
        async for ob in self.stream_order_book(symbol):
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
        """Poll funding rate updates for ``symbol`` via REST."""
        sym = normalize(symbol)
        if ":" not in symbol and "-" not in sym:
            raise NotImplementedError("Funding stream is not available for spot symbols")

        while True:
            data = await self.fetch_funding(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Poll open interest updates for ``symbol`` via REST."""
        sym = normalize(symbol)
        if ":" not in symbol and "-" not in sym:
            raise NotImplementedError("Open interest stream is not available for spot symbols")

        while True:
            data = await self.fetch_oi(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

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
        return await self.rest.create_order(symbol, type_, side, qty, price)

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        return await self.rest.cancel_order(order_id, symbol)
