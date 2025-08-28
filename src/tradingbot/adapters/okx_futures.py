# src/tradingbot/adapters/okx_futures.py
from __future__ import annotations

import asyncio
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
        maker_fee_bps: float | None = None,
        taker_fee_bps: float | None = None,
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

        self.rest = ccxt.okx(
            {
                "apiKey": key,
                "secret": secret,
                "password": passphrase,
                "enableRateLimit": True,
            }
        )
        self.rest.options["defaultType"] = "swap"
        self.rest.set_sandbox_mode(testnet)
        # Validar permisos disponibles
        validate_scopes(self.rest, log)

        self.ws_public_url = (
            "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
            if testnet
            else "wss://ws.okx.com:8443/ws/v5/public"
        )
        self.name = "okx_futures_testnet" if testnet else "okx_futures"
        self.maker_fee_bps = float(
            maker_fee_bps
            if maker_fee_bps is not None
            else (
                settings.okx_futures_testnet_maker_fee_bps
                if testnet
                else settings.okx_futures_maker_fee_bps
            )
        )
        self.taker_fee_bps = float(
            taker_fee_bps
            if taker_fee_bps is not None
            else (
                settings.okx_futures_testnet_taker_fee_bps
                if testnet
                else settings.okx_futures_taker_fee_bps
            )
        )

        self._start_fee_updater()

    async def update_fees(self, symbol: str | None = None) -> None:
        params = {"instType": "SWAP"}
        if symbol:
            params["instId"] = self.normalize_symbol(symbol)
        try:
            res = await self._request(self.rest.privateGetAccountTradeFee, params)
            data = (res.get("data") or [{}])[0]
            self.maker_fee_bps = abs(float(data.get("maker", 0.0))) * 10000
            self.taker_fee_bps = abs(float(data.get("taker", 0.0))) * 10000
        except Exception as e:  # pragma: no cover - best effort
            log.warning("update_fees failed: %s", e)

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

    def _normalize(self, symbol: str) -> str:
        """Return normalised symbol in OKX format."""

        return self.normalize_symbol(symbol).replace("/", "-").upper()

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        url = self.ws_public_url
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
        """Yield order book snapshots for ``symbol``.

        Events with incomplete bid or ask data are discarded.
        """
        url = self.ws_public_url
        sym = self._normalize(symbol)
        channel = self.DEPTH_TO_CHANNEL.get(depth)
        if channel is None:
            raise ValueError(f"depth must be one of {sorted(self.DEPTH_TO_CHANNEL)}")
        sub = {"op": "subscribe", "args": [{"channel": channel, "instId": sym}]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                if channel == "bbo-tbt":
                    bid_px_raw = d.get("bidPx")
                    ask_px_raw = d.get("askPx")
                    if bid_px_raw is None or ask_px_raw is None:
                        continue
                    bid_px = float(bid_px_raw)
                    ask_px = float(ask_px_raw)
                    if bid_px == 0 or ask_px == 0:
                        log.warning("bbo-tbt returned zero price; check channel permissions")
                        continue
                    bid_qty = float(d.get("bidSz") or 0)
                    ask_qty = float(d.get("askSz") or 0)
                    bids = [[bid_px, bid_qty]]
                    asks = [[ask_px, ask_qty]]
                else:
                    bids = [[float(p), float(q)] for p, q, *_ in d.get("bids", [])]
                    asks = [[float(p), float(q)] for p, q, *_ in d.get("asks", [])]
                ts = datetime.fromtimestamp(int(d.get("ts", 0)) / 1000, tz=timezone.utc)
                self.state.order_book[symbol] = {"bids": bids, "asks": asks}
                ob = self.normalize_order_book(symbol, ts, bids, asks)
                if len(ob["bid_px"]) != len(ob["bid_qty"]) or len(ob["ask_px"]) != len(ob["ask_qty"]):
                    continue
                yield ob
    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Emit best bid/ask quotes for ``symbol`` using books5 channel."""

        url = self.ws_public_url
        sym = self._normalize(symbol)
        sub = {"op": "subscribe", "args": [{"channel": "books5", "instId": sym}]}
        async for raw in self._ws_messages(url, json.dumps(sub)):
            msg = json.loads(raw)
            for d in msg.get("data", []) or []:
                bids = d.get("bids", [])
                asks = d.get("asks", [])
                if not bids or not asks:
                    continue
                bid_px = float(bids[0][0])
                bid_qty = float(bids[0][1])
                ask_px = float(asks[0][0])
                ask_qty = float(asks[0][1])
                if bid_px == 0 or ask_px == 0:
                    log.warning("books5 returned zero price; check channel permissions")
                    continue
                ts = datetime.fromtimestamp(int(d.get("ts", 0)) / 1000, tz=timezone.utc)
                yield {
                    "symbol": symbol,
                    "ts": ts,
                    "bid_px": bid_px,
                    "bid_qty": bid_qty,
                    "ask_px": ask_px,
                    "ask_qty": ask_qty,
                }


    async def stream_book_delta(self, symbol: str, depth: int = 5) -> AsyncIterator[dict]:
        """Yield incremental order book updates for ``symbol``.

        Events with incomplete bid or ask data are discarded.
        """
        prev: dict | None = None
        async for ob in self.stream_order_book(symbol, depth):
            bid_px = ob.get("bid_px", [])
            bid_qty = ob.get("bid_qty", [])
            ask_px = ob.get("ask_px", [])
            ask_qty = ob.get("ask_qty", [])
            if len(bid_px) != len(bid_qty) or len(ask_px) != len(ask_qty):
                continue
            curr_bids = list(zip(bid_px, bid_qty))
            curr_asks = list(zip(ask_px, ask_qty))
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

        OKX proporciona tanto el ``markPx`` como el ``indexPx`` en el ticker
        público.  Se consulta vía ``fetchTicker`` y se calcula la diferencia
        entre ambos valores.  Si la API o el adaptador REST no están
        disponibles se lanza ``NotImplementedError`` para dejar claro que el
        venue no soporta esta métrica.
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
