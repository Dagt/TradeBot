"""OKX connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook, Trade


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

    def _ws_trades_url(self, symbol: str) -> str:
        return "wss://ws.okx.com:8443/ws/v5/public"

    def _ws_trades_subscribe(self, symbol: str) -> str:
        return json.dumps({"op": "subscribe", "args": [{"channel": "trades", "instId": symbol}]})

    def _parse_trade(self, msg: str, symbol: str) -> Trade:
        data = json.loads(msg)
        trade = (data.get("data") or [{}])[0]
        ts = trade.get("ts") or trade.get("t") or 0
        if ts > 1e12:
            ts = int(ts) / 1000
        else:
            ts = int(ts)
        return Trade(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            price=float(trade.get("px", 0.0)),
            amount=float(trade.get("sz", 0.0)),
            side=trade.get("side", ""),
        )
