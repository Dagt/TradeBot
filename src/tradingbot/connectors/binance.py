"""Binance connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook, Trade


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

    def _ws_trades_url(self, symbol: str) -> str:
        return f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"

    def _ws_trades_subscribe(self, symbol: str) -> str:
        return json.dumps(
            {"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@trade"], "id": 1}
        )

    def _parse_trade(self, msg: str, symbol: str) -> Trade:
        data = json.loads(msg)
        ts = data.get("T") or data.get("t") or 0
        if ts > 1e12:
            ts /= 1000
        return Trade(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            price=float(data.get("p", 0.0)),
            amount=float(data.get("q", 0.0)),
            side="sell" if data.get("m") else "buy",
        )
