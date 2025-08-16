"""Binance connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook


class BinanceConnector(ExchangeConnector):
    name = "binance"

    def _ws_url(self, symbol: str) -> str:
        return f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth"

    def _ws_subscribe(self, symbol: str) -> str:
        return json.dumps(
            {"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@depth"], "id": 1}
        )

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:
        data = json.loads(msg)
        bids = [(float(p), float(q)) for p, q in data.get("b", [])]
        asks = [(float(p), float(q)) for p, q in data.get("a", [])]
        return OrderBook(
            timestamp=datetime.utcnow(),
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )
