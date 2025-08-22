# src/tradingbot/adapters/binance_futures_ws.py
import asyncio
import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

from .base import ExchangeAdapter
from ..config import settings
from ..core.symbols import normalize

log = logging.getLogger(__name__)

def _stream_name(symbol: str, channel: str = "aggTrade") -> str:
    # futures usa minúsculas y sin "/"
    return symbol.replace("/", "").lower() + f"@{channel}"

class BinanceFuturesWSAdapter(ExchangeAdapter):
    """
    WS de Binance USDⓈ-M Futures (UM) para datos de mercado.
    Permite operar tanto en mainnet como en testnet.
    """
    name = "binance_futures_um_testnet_ws"

    def __init__(
        self,
        ws_base: str | None = None,
        rest: ExchangeAdapter | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = False,
    ):
        super().__init__()
        self._testnet = testnet
        default_ws = (
            "wss://stream.binancefuture.com/stream?streams="
            if testnet
            else "wss://fstream.binance.com/stream?streams="
        )
        self.ws_base = ws_base or default_ws
        self.rest = rest
        self._api_key = api_key
        self._api_secret = api_secret

    def _ensure_rest(self):
        if self.rest is None:
            from .binance_futures import BinanceFuturesAdapter

            self.rest = BinanceFuturesAdapter(
                api_key=self._api_key or settings.binance_futures_api_key,
                api_secret=self._api_secret or settings.binance_futures_api_secret,
                testnet=self._testnet,
            )

    async def stream_trades_multi(self, symbols: Iterable[str], channel: str = "aggTrade") -> AsyncIterator[dict]:
        streams = "/".join(_stream_name(normalize(s), channel) for s in symbols)
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
        stream = _stream_name(normalize(symbol), f"depth{depth}@100ms")
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

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Stream best bid/ask updates using the ``bookTicker`` feed."""

        stream = _stream_name(normalize(symbol), "bookTicker")
        async for raw in self._ws_messages(self.ws_base + stream):
            d = json.loads(raw).get("data") or {}
            yield {
                "symbol": symbol,
                "ts": datetime.fromtimestamp(int(d.get("T", 0)) / 1000, tz=timezone.utc),
                "bid_px": float(d.get("b") or 0.0),
                "bid_qty": float(d.get("B") or 0.0),
                "ask_px": float(d.get("a") or 0.0),
                "ask_qty": float(d.get("A") or 0.0),
            }

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Yield order book deltas relative to the previous snapshot."""

        prev: dict | None = None
        async for ob in self.stream_order_book(symbol, depth):
            curr_bids = list(zip(ob.get("bid_px", []), ob.get("bid_qty", [])))
            curr_asks = list(zip(ob.get("ask_px", []), ob.get("ask_qty", [])))
            if prev is None:
                delta_bids = curr_bids
                delta_asks = curr_asks
            else:
                prev_bids = dict(zip(prev.get("bid_px", []), prev.get("bid_qty", [])))
                prev_asks = dict(zip(prev.get("ask_px", []), prev.get("ask_qty", [])))
                delta_bids = [[p, q] for p, q in curr_bids if prev_bids.get(p) != q]
                delta_bids += [[p, 0.0] for p in prev_bids.keys() - {p for p, _ in curr_bids}]
                delta_asks = [[p, q] for p, q in curr_asks if prev_asks.get(p) != q]
                delta_asks += [[p, 0.0] for p in prev_asks.keys() - {p for p, _ in curr_asks}]
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
        """Stream funding rate updates for ``symbol``."""

        stream = _stream_name(normalize(symbol), "markPrice@1s")
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or msg
            rate = d.get("r") or d.get("fundingRate")
            if rate is None:
                continue
            ts_ms = d.get("E") or d.get("T") or 0
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            index_px = float(d.get("i") or 0.0)
            yield {
                "symbol": symbol,
                "ts": ts,
                "rate": float(rate),
                "index_px": index_px,
            }

    async def stream_open_interest(self, symbol: str, interval: str = "1m") -> AsyncIterator[dict]:
        """Stream open interest updates for ``symbol``.

        Parameters
        ----------
        symbol: str
            Market symbol to subscribe to.
        interval: str, optional
            Update interval for open interest aggregation.  Binance supports
            values such as ``"1m"`` and ``"5m"``.  Defaults to ``"1m"``.

        Notes
        -----
        The Binance Futures testnet does **not** currently support the
        ``openInterest`` websocket channel.  When the adapter is running in
        testnet mode a :class:`NotImplementedError` is raised to make this
        limitation explicit.
        """

        if self._testnet:
            raise NotImplementedError("openInterest stream not available on Binance Futures testnet")

        stream = _stream_name(normalize(symbol), f"openInterest@{interval}")
        url = self.ws_base + stream

        messages = self._ws_messages(url)
        first = True
        while True:
            try:
                raw = await asyncio.wait_for(messages.__anext__(), timeout=15)
            except asyncio.TimeoutError:
                log.warning("No message received on %s for 15s", stream)
                messages = self._ws_messages(url)
                continue
            except StopAsyncIteration:
                return

            if first:
                log.debug("Subscribed to %s", stream)
                first = False

            msg = json.loads(raw)
            d = msg.get("data") or msg
            oi = d.get("oi") or d.get("openInterest") or d.get("o")
            if oi is None:
                continue
            ts_ms = d.get("E") or d.get("T") or 0
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            yield {"symbol": symbol, "ts": ts, "oi": float(oi)}

    async def fetch_funding(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_funding(symbol)
        sym = normalize(symbol)
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

        sym = normalize(symbol)
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
        sym = normalize(symbol)
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
