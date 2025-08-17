"""Binance connector using CCXT REST and native websockets."""
from __future__ import annotations

import json
from datetime import datetime

from .base import (
    ExchangeConnector,
    Funding,
    OpenInterest,
    OrderBook,
    Trade,
)


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

    async def fetch_funding(self, symbol: str) -> Funding:
        """Fetch current funding rate for ``symbol``."""

        method = getattr(self.rest, "fetch_funding_rate", None) or getattr(
            self.rest, "fetchFundingRate", None
        )
        if method is None:
            raise NotImplementedError("Funding not supported")
        data = await self._rest_call(method, symbol)
        ts = (
            data.get("timestamp")
            or data.get("fundingTime")
            or data.get("time")
            or data.get("ts")
            or 0
        )
        if ts > 1e12:
            ts /= 1000
        return Funding(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            rate=float(
                data.get("fundingRate")
                or data.get("rate")
                or data.get("value")
                or 0.0
            ),
        )

    async def fetch_open_interest(self, symbol: str) -> OpenInterest:
        """Fetch current open interest for ``symbol``."""

        method = getattr(self.rest, "fapiPublic_get_open_interest", None) or getattr(
            self.rest, "fapiPublicGetOpenInterest", None
        )
        if method is None:
            raise NotImplementedError("Open interest not supported")
        sym = symbol.replace("/", "")
        data = await self._rest_call(method, {"symbol": sym})
        ts = data.get("time") or data.get("timestamp") or data.get("ts") or 0
        if ts > 1e12:
            ts /= 1000
        return OpenInterest(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            oi=float(
                data.get("openInterest")
                or data.get("oi")
                or data.get("value")
                or 0.0
            ),
        )
