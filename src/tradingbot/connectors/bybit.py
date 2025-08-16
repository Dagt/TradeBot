"""Bybit connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook, Trade


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

    def _ws_trades_url(self, symbol: str) -> str:
        # Public spot endpoint
        return "wss://stream.bybit.com/v5/public/spot"

    def _ws_trades_subscribe(self, symbol: str) -> str:
        return json.dumps({"op": "subscribe", "args": [f"publicTrade.{symbol}"]})

    def _parse_trade(self, msg: str, symbol: str) -> Trade:
        data = json.loads(msg)
        trade_data = (data.get("data") or [{}])[0]
        ts = trade_data.get("t") or trade_data.get("T") or 0
        if ts > 1e12:
            ts /= 1000
        return Trade(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            price=float(trade_data.get("p", 0.0)),
            amount=float(trade_data.get("v", 0.0)),
            side=(trade_data.get("S") or trade_data.get("side", "")).lower(),
        )
