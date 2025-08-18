from __future__ import annotations

import os
from datetime import datetime
from typing import List

import httpx

from .base import OrderBook, Trade


class KaikoConnector:
    """Simple Kaiko REST connector.

    Examples
    --------
    >>> c = KaikoConnector("KEY")
    >>> ob = await c.fetch_order_book("binance", "btc-usdt")

    Limitations
    -----------
    Only historical and recent public data are accessible. Rate limits and
    data availability depend on the Kaiko subscription plan.
    """

    name = "kaiko"
    _BASE_URL = "https://us.market-api.kaiko.io/v2/data"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("KAIKO_API_KEY", "")

    async def fetch_trades(
        self, exchange: str, pair: str, limit: int = 100
    ) -> List[Trade]:
        """Fetch recent trades for *pair* on *exchange*.

        Parameters match Kaiko's "recent trades" endpoint. Only a subset of the
        response fields is mapped to :class:`Trade` objects.
        """

        url = f"{self._BASE_URL}/trades.v1/exchanges/{exchange}/spot/{pair}/recent"
        headers = {"X-Api-Key": self.api_key}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={"limit": limit}, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", [])

        trades: List[Trade] = []
        for t in data:
            ts = t.get("timestamp") or t.get("time") or ""
            # Kaiko timestamps are ISO8601 strings
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
            trades.append(
                Trade(
                    timestamp=dt,
                    exchange=exchange,
                    symbol=pair,
                    price=float(t.get("price", 0.0)),
                    amount=float(t.get("amount", 0.0)),
                    side=t.get("taker_side", ""),
                )
            )
        return trades

    async def fetch_order_book(
        self, exchange: str, pair: str, depth: int = 10
    ) -> OrderBook:
        """Fetch a snapshot of the order book."""

        url = (
            f"{self._BASE_URL}/order_book.snapshots.v1/exchanges/{exchange}/spot/{pair}/books/latest"
        )
        headers = {"X-Api-Key": self.api_key}
        params = {"depth": depth}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json().get("data", {})

        bids_raw = payload.get("bids", [])[:depth]
        asks_raw = payload.get("asks", [])[:depth]
        ts = payload.get("timestamp", "")
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
        bids = [(float(p), float(q)) for p, q in bids_raw]
        asks = [(float(p), float(q)) for p, q in asks_raw]
        return OrderBook(timestamp=dt, exchange=exchange, symbol=pair, bids=bids, asks=asks)
