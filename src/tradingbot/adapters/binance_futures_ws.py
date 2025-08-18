# src/tradingbot/adapters/binance_futures_ws.py
import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

from .base import ExchangeAdapter
from ..config import settings

log = logging.getLogger(__name__)

def _stream_name(symbol: str, channel: str = "aggTrade") -> str:
    # futures usa minúsculas y sin "/"
    return symbol.replace("/", "").lower() + f"@{channel}"

class BinanceFuturesWSAdapter(ExchangeAdapter):
    """
    WS de Binance USDⓈ-M Futures **TESTNET** (UM) para datos de mercado.
    """
    name = "binance_futures_um_testnet_ws"

    def __init__(
        self,
        ws_base: str | None = None,
        rest: ExchangeAdapter | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool | None = None,
    ):
        super().__init__()
        # UM testnet combined streams:
        #   wss://stream.binancefuture.com/stream?streams=btcusdt@aggTrade/ethusdt@aggTrade
        self.ws_base = ws_base or "wss://stream.binancefuture.com/stream?streams="
        self.rest = rest
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

    def _ensure_rest(self):
        if self.rest is None:
            from .binance_futures import BinanceFuturesAdapter

            self.rest = BinanceFuturesAdapter(
                api_key=self._api_key or settings.binance_futures_api_key,
                api_secret=self._api_secret or settings.binance_futures_api_secret,
                testnet=self._testnet
                if self._testnet is not None
                else settings.binance_futures_testnet,
            )

    async def stream_trades_multi(self, symbols: Iterable[str], channel: str = "aggTrade") -> AsyncIterator[dict]:
        streams = "/".join(_stream_name(self.normalize_symbol(s), channel) for s in symbols)
        url = self.ws_base + streams
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            stream = (msg.get("stream") or "")
            d = msg.get("data") or {}
            price = d.get("p") or d.get("price")
            qty = d.get("q")
            ts_ms = d.get("T") or d.get("E")
            if price is None:
                continue
            symbol_key = stream.split("@")[0].upper()
            base = symbol_key[:-4]
            symbol = f"{base}/USDT"

            price = float(price)
            qty = float(qty or 0.0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            side = "sell" if d.get("m") else "buy"
            self.state.last_px[symbol] = price
            yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        async for t in self.stream_trades_multi([symbol]):
            yield t

    async def stream_order_book(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        stream = _stream_name(self.normalize_symbol(symbol), f"depth{depth}@100ms")
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or msg
            bids = d.get("b") or d.get("bids") or []
            asks = d.get("a") or d.get("asks") or []
            ts_ms = d.get("T") or d.get("E")
            bids_n = [[float(b[0]), float(b[1])] for b in bids]
            asks_n = [[float(a[0]), float(a[1])] for a in asks]
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            self.state.order_book[symbol] = {"bids": bids_n, "asks": asks_n}
            yield self.normalize_order_book(symbol, ts, bids_n, asks_n)

    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_funding(symbol)
        sym = self.normalize_symbol(symbol)
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"

        def _fetch():
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())

        return await self._request(_fetch)

    async def fetch_basis(self, symbol: str):
        """Return the basis (mark - index) for ``symbol``.

        The futures premium index endpoint provides both the mark and index
        price.  If a REST adapter is configured we simply delegate to it,
        otherwise we query the public endpoint directly using ``urllib``.
        The response fields are normalised into a ``{"ts": datetime,
        "basis": float}`` mapping.
        """

        if self.rest:
            return await self.rest.fetch_basis(symbol)

        sym = self.normalize_symbol(symbol)
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"

        def _fetch():
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())

        data = await self._request(_fetch)
        ts_raw = data.get("time") or data.get("timestamp") or data.get("ts") or 0
        ts_raw = int(ts_raw)
        if ts_raw > 1e12:
            ts_raw //= 1000
        ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        mark_px = float(data.get("markPrice") or data.get("mark_price") or 0.0)
        index_px = float(data.get("indexPrice") or data.get("index_price") or 0.0)
        return {"ts": ts, "basis": mark_px - index_px}

    async def fetch_oi(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_oi(symbol)
        sym = self.normalize_symbol(symbol)
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}"

        def _fetch():
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())

        return await self._request(_fetch)

    # interfaz ExchangeAdapter (no aplica envío por WS)
    async def place_order(self, *args, **kwargs):
        self._ensure_rest()
        return await self.rest.place_order(*args, **kwargs)

    async def cancel_order(self, order_id: str, *args, **kwargs):
        self._ensure_rest()
        return await self.rest.cancel_order(order_id, *args, **kwargs)
