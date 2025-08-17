from __future__ import annotations

import os
from datetime import datetime
from typing import List

import httpx

from .base import OrderBook, Trade


class CoinAPIConnector:
    """REST connector for CoinAPI."""

    name = "coinapi"
    _BASE_URL = "https://rest.coinapi.io/v1"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("COINAPI_KEY", "")

    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        url = f"{self._BASE_URL}/trades/{symbol}"
        headers = {"X-CoinAPI-Key": self.api_key}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={"limit": limit}, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        trades: List[Trade] = []
        for t in data:
            ts = t.get("time_exchange") or t.get("time_trade") or ""
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
            trades.append(
                Trade(
                    timestamp=dt,
                    exchange=self.name,
                    symbol=symbol,
                    price=float(t.get("price", 0.0)),
                    amount=float(t.get("size", 0.0)),
                    side=t.get("taker_side", ""),
                )
            )
        return trades

    async def fetch_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        url = f"{self._BASE_URL}/orderbooks/current/{symbol}"
        headers = {"X-CoinAPI-Key": self.api_key}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        bids_raw = payload.get("bids", [])[:depth]
        asks_raw = payload.get("asks", [])[:depth]
        ts = payload.get("time_exchange") or payload.get("time_coinapi") or ""
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
        bids = [(float(b.get("price")), float(b.get("size"))) for b in bids_raw]
        asks = [(float(a.get("price")), float(a.get("size"))) for a in asks_raw]
        return OrderBook(timestamp=dt, exchange=self.name, symbol=symbol, bids=bids, asks=asks)
