from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Iterable, List

import httpx

from .base import Funding, OpenInterest, OrderBook, Trade


class CoinAPIConnector:
    """REST connector for CoinAPI.

    Examples
    --------
    >>> c = CoinAPIConnector("KEY")
    >>> trades = await c.fetch_trades("BTCUSD")

    Limitations
    -----------
    Only public market data is available; authentication is required via an
    API key and the service enforces strict rate limits.
    """

    name = "coinapi"
    _BASE_URL = "https://rest.coinapi.io/v1"

    def __init__(self, api_key: str | None = None, rate_limit: int = 5) -> None:
        self.api_key = api_key or os.getenv("COINAPI_KEY", "")
        if not self.api_key:
            raise ValueError("COINAPI_KEY missing")
        self._sem = asyncio.Semaphore(rate_limit)

    async def fetch_trades(
        self,
        symbol: str,
        limit: int = 100,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> List[Trade]:
        url = f"{self._BASE_URL}/trades/{symbol}"
        headers = {"X-CoinAPI-Key": self.api_key}
        params: dict[str, Any] = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
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

    async def fetch_order_book(
        self,
        symbol: str,
        depth: int = 10,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> OrderBook:
        url = f"{self._BASE_URL}/orderbooks/current/{symbol}"
        headers = {"X-CoinAPI-Key": self.api_key}
        params: dict[str, Any] = {}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        bids_raw = payload.get("bids", [])[:depth]
        asks_raw = payload.get("asks", [])[:depth]
        ts = payload.get("time_exchange") or payload.get("time_coinapi") or ""
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
        bids = [(float(b.get("price")), float(b.get("size"))) for b in bids_raw]
        asks = [(float(a.get("price")), float(a.get("size"))) for a in asks_raw]
        return OrderBook(timestamp=dt, exchange=self.name, symbol=symbol, bids=bids, asks=asks)

    async def _paginate(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
        max_pages: int | None = None,
    ) -> Iterable[dict]:
        """Generic pagination helper for CoinAPI REST endpoints."""

        results: list[dict] = []
        page = 0
        async with httpx.AsyncClient() as client:
            next_time: str | None = None
            while True:
                req_params = params.copy()
                if next_time:
                    req_params["start_time"] = next_time
                async with self._sem:
                    resp = await client.get(url, params=req_params, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                items = payload.get("data") if isinstance(payload, dict) else payload
                results.extend(items)
                next_time = payload.get("next_time") if isinstance(payload, dict) else None
                page += 1
                if not next_time or (max_pages and page >= max_pages):
                    break
        return results

    async def fetch_funding(
        self,
        symbol: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> List[Funding]:
        """Fetch historical funding rates for *symbol* using CoinAPI."""

        url = f"{self._BASE_URL}/futures/funding_rates/{symbol}"
        headers = {"X-CoinAPI-Key": self.api_key}
        params: dict[str, Any] = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        raw = await self._paginate(url, params, headers, max_pages)
        records: list[Funding] = []
        for f in raw:
            ts = f.get("time_period_start") or f.get("time") or f.get("timestamp") or ""
            dt = (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, str) and ts
                else datetime.utcnow()
            )
            records.append(
                Funding(
                    timestamp=dt,
                    exchange=self.name,
                    symbol=symbol,
                    rate=float(
                        f.get("funding_rate")
                        or f.get("rate")
                        or f.get("value")
                        or 0.0
                    ),
                )
            )
        return records

    async def fetch_open_interest(
        self,
        symbol: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> List[OpenInterest]:
        """Fetch historical open interest using CoinAPI."""

        url = f"{self._BASE_URL}/futures/open_interest/{symbol}"
        headers = {"X-CoinAPI-Key": self.api_key}
        params: dict[str, Any] = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        raw = await self._paginate(url, params, headers, max_pages)
        records: list[OpenInterest] = []
        for r in raw:
            ts = (
                r.get("time_period_start")
                or r.get("time")
                or r.get("timestamp")
                or ""
            )
            dt = (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, str) and ts
                else datetime.utcnow()
            )
            records.append(
                OpenInterest(
                    timestamp=dt,
                    exchange=self.name,
                    symbol=symbol,
                    oi=float(
                        r.get("open_interest")
                        or r.get("openInterest")
                        or r.get("oi")
                        or r.get("value")
                        or 0.0
                    ),
                )
            )
        return records

    fetch_oi = fetch_open_interest
