"""Bybit connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook


class BybitConnector(ExchangeConnector):
    name = "bybit"

    def _ws_url(self, symbol: str) -> str:
        # Public spot endpoint
        return "wss://stream.bybit.com/v5/public/spot"

    def _ws_subscribe(self, symbol: str) -> str:
        return json.dumps({"op": "subscribe", "args": [f"orderbook.50.{symbol}"]})

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:
        data = json.loads(msg)
        ob = data.get("data", {})
        bids = [(float(p), float(q)) for p, q in ob.get("b", ob.get("bids", []))]
        asks = [(float(p), float(q)) for p, q in ob.get("a", ob.get("asks", []))]
        return OrderBook(
            timestamp=datetime.utcnow(),
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )
