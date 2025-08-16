"""OKX connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook


class OKXConnector(ExchangeConnector):
    name = "okx"

    def _ws_url(self, symbol: str) -> str:
        return "wss://ws.okx.com:8443/ws/v5/public"

    def _ws_subscribe(self, symbol: str) -> str:
        return json.dumps({"op": "subscribe", "args": [{"channel": "books", "instId": symbol}]})

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:
        data = json.loads(msg)
        arg_data = data.get("data", [{}])[0]
        bids = [(float(p), float(q)) for p, q in arg_data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in arg_data.get("asks", [])]
        return OrderBook(
            timestamp=datetime.utcnow(),
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )
