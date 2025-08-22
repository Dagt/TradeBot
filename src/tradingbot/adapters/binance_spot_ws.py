# src/tradingbot/adapters/binance_spot_ws.py
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

from .base import ExchangeAdapter
from ..core.symbols import normalize

log = logging.getLogger(__name__)


def _stream_name(symbol: str, channel: str = "trade") -> str:
    return symbol.replace("/", "").lower() + f"@{channel}"

class BinanceSpotWSAdapter(ExchangeAdapter):
    """
    WS de Binance Spot para trades.
    Usa el entorno de *testnet* si ``testnet=True``.
    """
    name = "binance_spot_ws"

    def __init__(
        self,
        ws_base: str | None = None,
        rest: ExchangeAdapter | None = None,
        testnet: bool = False,
    ):
        super().__init__()
        self.rest = rest
        self.testnet = testnet

        default_ws_base = (
            "wss://testnet.binance.vision/stream?streams="
            if testnet
            else "wss://stream.binance.com:9443/stream?streams="
        )
        self.ws_base = ws_base or default_ws_base

        if testnet:
            self.name = "binance_spot_testnet_ws"

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        stream = _stream_name(normalize(symbol))
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or {}
            price = d.get("p")
            qty = d.get("q")
            ts_ms = d.get("T")
            if price is None:
                continue
            price = float(price)
            qty = float(qty or 0.0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            side = "sell" if d.get("m") else "buy"
            self.state.last_px[symbol] = price
            yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_trades_multi(self, symbols: Iterable[str]) -> AsyncIterator[dict]:
        """
        Un solo socket con múltiples streams. Yields:
        {"symbol": <sym>, "ts": datetime, "price": float, "qty": float, "side": "buy"/"sell"}
        """
        streams = "/".join(_stream_name(normalize(s)) for s in symbols)
        url = self.ws_base + streams
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            stream = (msg.get("stream") or "")
            data = msg.get("data") or {}
            price = data.get("p")
            qty = data.get("q")
            ts_ms = data.get("T")
            if price is None:
                continue
            symbol_key = stream.split("@")[0].upper()
            if symbol_key.endswith("USDT"):
                base = symbol_key[:-4]
                symbol = f"{base}/USDT"
            else:
                candidates = ("USDT","BUSD","BTC","ETH","BNB","FDUSD","TUSD")
                match = next((q for q in candidates if symbol_key.endswith(q)), None)
                symbol = f"{symbol_key[:-len(match)]}/{match}" if match else symbol_key

            price = float(price)
            qty = float(qty or 0.0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            side = "sell" if data.get("m") else "buy"
            self.state.last_px[symbol] = price
            yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_order_book(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        stream = _stream_name(normalize(symbol), f"depth{depth}@100ms")
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or msg
            ts_ms = d.get("T") or d.get("E")
            bids = d.get("bids") or d.get("b") or []
            asks = d.get("asks") or d.get("a") or []
            bids_n = [[float(b[0]), float(b[1])] for b in bids]
            asks_n = [[float(a[0]), float(a[1])] for a in asks]
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            self.state.order_book[symbol] = {"bids": bids_n, "asks": asks_n}
            yield self.normalize_order_book(symbol, ts, bids_n, asks_n)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Stream best bid/ask for ``symbol`` using Binance's ``bookTicker``."""

        stream = _stream_name(normalize(symbol), "bookTicker")
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or msg
            bid = d.get("b")
            bid_qty = d.get("B")
            ask = d.get("a")
            ask_qty = d.get("A")
            ts_ms = d.get("T") or d.get("E")
            ts = (
                datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if ts_ms
                else datetime.now(timezone.utc)
            )
            yield {
                "symbol": symbol,
                "ts": ts,
                "bid_px": float(bid) if bid is not None else None,
                "bid_qty": float(bid_qty or 0.0),
                "ask_px": float(ask) if ask is not None else None,
                "ask_qty": float(ask_qty or 0.0),
            }

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Stream order book deltas compared to the previous snapshot."""

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

    async def stream_funding(self, symbol: str) -> AsyncIterator[dict]:  # pragma: no cover - not supported
        raise NotImplementedError("Funding stream not supported for Spot WS")

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:  # pragma: no cover - not supported
        raise NotImplementedError("Open interest stream not supported for Spot WS")

    async def fetch_funding(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_funding(symbol)
        raise NotImplementedError("Funding no disponible para mercados Spot")

    async def fetch_basis(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_basis(symbol)
        raise NotImplementedError("Basis no disponible para mercados Spot")

    async def fetch_oi(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_oi(symbol)
        raise NotImplementedError("Open interest no disponible para Spot")

    async def place_order(self, *args, **kwargs) -> dict:
        if self.rest:
            return await self.rest.place_order(*args, **kwargs)
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str, *args, **kwargs) -> dict:
        if self.rest:
            return await self.rest.cancel_order(order_id, *args, **kwargs)
        raise NotImplementedError("no aplica en WS")
