"""Deribit connector leveraging CCXT and native websockets.

Examples
--------
>>> c = DeribitConnector()
>>> trade = await c.stream_trades("BTC-PERPETUAL").__anext__()

Limitations
-----------
Only public market data is implemented; order management is not available
via this connector.
"""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook, Trade


class DeribitConnector(ExchangeConnector):
    name = "deribit"

    def _ws_url(self, symbol: str) -> str:
        return "wss://www.deribit.com/ws/api/v2"

    def _ws_subscribe(self, symbol: str) -> str:
        chan = f"book.{symbol}.raw"
        return json.dumps({
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [chan]},
            "id": 1,
        })

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:
        data = json.loads(msg)
        params = data.get("params", {})
        book = params.get("data", {})
        bids = [(float(p), float(q)) for p, q, *_ in book.get("bids", [])]
        asks = [(float(p), float(q)) for p, q, *_ in book.get("asks", [])]
        ts = book.get("timestamp") or 0
        if ts > 1e12:
            ts /= 1000
        return OrderBook(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )

    def _ws_trades_url(self, symbol: str) -> str:
        return "wss://www.deribit.com/ws/api/v2"

    def _ws_trades_subscribe(self, symbol: str) -> str:
        chan = f"trades.{symbol}.raw"
        return json.dumps({
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": [chan]},
            "id": 1,
        })

    def _parse_trade(self, msg: str, symbol: str) -> Trade:
        data = json.loads(msg)
        params = data.get("params", {})
        trades = params.get("data") or []
        t = trades[0] if trades else {}
        ts = t.get("timestamp") or t.get("time") or 0
        if ts > 1e12:
            ts /= 1000
        return Trade(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            price=float(t.get("price", 0.0)),
            amount=float(t.get("amount", 0.0)),
            side=str(t.get("direction", t.get("side", ""))).lower(),
        )
