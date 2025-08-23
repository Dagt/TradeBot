"""WebSocket adapter for streaming markets from Deribit."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from .base import ExchangeAdapter
from .deribit import DeribitAdapter

log = logging.getLogger(__name__)


def normalize(symbol: str) -> str:
    """Return Deribit instrument name for ``symbol``.

    The websocket adapter shares the symbol mapping with the REST adapter so
    users can pass generic pairs like ``ETHUSDT`` and obtain the venue specific
    instrument ``ETH-PERPETUAL``.
    """

    return DeribitAdapter.normalize(symbol)


class DeribitWSAdapter(ExchangeAdapter):
    """Websocket wrapper delegando a :class:`DeribitAdapter` para REST."""

    name = "deribit_futures_ws"
    # only websocket-native streams are advertised; funding and open interest
    # require a REST fallback and are therefore omitted from ``supported_kinds``
    supported_kinds = ["trades", "orderbook", "bba", "delta"]

    def __init__(self, rest: DeribitAdapter | None = None, testnet: bool = False):
        super().__init__()
        self.rest = rest
        self.ws_url = (
            "wss://test.deribit.com/ws/api/v2" if testnet else "wss://www.deribit.com/ws/api/v2"
        )
        self.name = "deribit_futures_ws_testnet" if testnet else "deribit_futures_ws"

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Map generic pairs to Deribit instrument names."""

        return normalize(symbol)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Stream trades from Deribit public websocket.

        Suscribe al canal ``trades.{symbol}`` y emite cada transacción
        con ``price``, ``amount`` y ``ts``. Reintenta en caso de
        desconexión.
        """

        sym = self._normalize(symbol)
        channel = f"trades.{sym}.100ms"
        sub = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [channel]},
            "id": 1,
        }

        while True:
            try:
                async for raw in self._ws_messages(self.ws_url, json.dumps(sub)):
                    msg = json.loads(raw)
                    if msg.get("error"):
                        err = msg.get("error")
                        log.error("WS subscription error for %s: %s", channel, err)
                        raise ValueError(f"WS subscription error for {channel}: {err}")
                    params = msg.get("params") or {}
                    if params.get("channel") != channel:
                        raise ValueError(
                            f"Unexpected channel {params.get('channel')} for {channel}"
                        )
                    data = params.get("data") or []
                    if not data:
                        continue
                    for t in data:
                        price = float(t.get("price") or 0.0)
                        qty = float(t.get("amount") or t.get("size") or 0.0)
                        side = str(t.get("direction") or t.get("side") or "").lower()
                        ts_ms = int(t.get("timestamp") or t.get("time") or 0)
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                        self.state.last_px[sym] = price
                        yield self.normalize_trade(sym, ts, price, qty, side)
            except asyncio.CancelledError:
                raise
            except ValueError:
                raise
            except Exception as e:  # pragma: no cover - reconnection path
                log.warning("WS trades disconnected (%s), reconnecting ...", e)
                await asyncio.sleep(1.0)

    ORDER_BOOK_DEPTHS = {1, 5, 10, 20, 30, 50}

    async def stream_order_book(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Stream order book snapshots from Deribit."""

        if depth not in self.ORDER_BOOK_DEPTHS:
            raise ValueError(f"depth must be one of {sorted(self.ORDER_BOOK_DEPTHS)}")

        sym = self._normalize(symbol)
        if self.rest:
            try:
                async for ob in self.rest.stream_order_book(sym, depth):
                    yield ob
                return
            except (NotImplementedError, AttributeError):
                pass

        channel = f"book.{sym}.none.{depth}.100ms"
        sub = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [channel]},
            "id": 1,
        }

        async for raw in self._ws_messages(self.ws_url, json.dumps(sub)):
            msg = json.loads(raw)
            if msg.get("error"):
                log.error("WS subscription error: %s", msg)
                continue
            params = msg.get("params") or {}
            if params.get("channel") != channel:
                raise ValueError(
                    f"Unexpected channel {params.get('channel')} for {channel}"
                )
            data = params.get("data") or {}
            if not data:
                continue
            bids = [[float(p), float(q)] for p, q, *_ in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q, *_ in data.get("asks", [])]
            ts_ms = int(data.get("timestamp") or data.get("ts") or 0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            self.state.order_book[sym] = {"bids": bids, "asks": asks}
            yield self.normalize_order_book(sym, ts, bids, asks)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Stream best bid/ask levels for ``symbol``."""

        sym = self._normalize(symbol)
        async for ob in self.stream_order_book(sym, depth=1):
            bid_px = ob.get("bid_px", [])
            ask_px = ob.get("ask_px", [])
            bid_qty = ob.get("bid_qty", [])
            ask_qty = ob.get("ask_qty", [])
            yield {
                "symbol": sym,
                "ts": ob.get("ts"),
                "bid_px": bid_px[0] if bid_px else None,
                "bid_qty": bid_qty[0] if bid_qty else 0.0,
                "ask_px": ask_px[0] if ask_px else None,
                "ask_qty": ask_qty[0] if ask_qty else 0.0,
            }

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Stream order book deltas relative to previous snapshots."""

        sym = self._normalize(symbol)
        prev: dict | None = None
        async for ob in self.stream_order_book(sym, depth):
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
                "symbol": sym,
                "ts": ob.get("ts"),
                "bid_px": [p for p, _ in delta_bids],
                "bid_qty": [q for _, q in delta_bids],
                "ask_px": [p for p, _ in delta_asks],
                "ask_qty": [q for _, q in delta_asks],
            }

    async def stream_funding(self, symbol: str) -> AsyncIterator[dict]:
        if not self.rest:
            raise NotImplementedError("Funding stream requires REST adapter")
        sym = self._normalize(symbol)
        while True:
            data = await self.fetch_funding(sym)
            yield {"symbol": sym, **data}
            await asyncio.sleep(60)

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        if not self.rest:
            raise NotImplementedError("Open interest stream requires REST adapter")
        sym = self._normalize(symbol)
        while True:
            data = await self.fetch_oi(sym)
            yield {"symbol": sym, **data}
            await asyncio.sleep(60)

    async def fetch_funding(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_funding(symbol)
        raise NotImplementedError("Funding no disponible")

    async def fetch_basis(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_basis(symbol)
        raise NotImplementedError("Basis no disponible")

    async def fetch_oi(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_oi(symbol)
        raise NotImplementedError("Open interest no disponible")

    fetch_open_interest = fetch_oi

    async def place_order(self, *args, **kwargs) -> dict:
        if self.rest:
            return await self.rest.place_order(*args, **kwargs)
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str) -> dict:
        if self.rest:
            return await self.rest.cancel_order(order_id)
        raise NotImplementedError("no aplica en WS")
