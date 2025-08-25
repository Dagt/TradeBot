from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Iterable, List

import httpx

from .base import Funding, OpenInterest, OrderBook, Trade


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

    def __init__(self, api_key: str | None = None, rate_limit: int = 5) -> None:
        self.api_key = api_key or os.getenv("KAIKO_API_KEY", "")
        if not self.api_key:
            raise ValueError("KAIKO_API_KEY missing")
        # very small semaphore based limiter for historical downloads
        self._sem = asyncio.Semaphore(rate_limit)

    async def fetch_trades(
        self,
        exchange: str,
        pair: str,
        limit: int = 100,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> List[Trade]:
        """Fetch recent trades for *pair* on *exchange*.

        Optional ``start_time`` and ``end_time`` parameters (ISO8601 strings) are
        forwarded to the Kaiko REST API allowing constrained historical
        downloads when supported by the endpoint.
        """

        url = f"{self._BASE_URL}/trades.v1/exchanges/{exchange}/spot/{pair}/recent"
        headers = {"X-Api-Key": self.api_key}
        params: dict[str, Any] = {"limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
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
        self,
        exchange: str,
        pair: str,
        depth: int = 10,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> OrderBook:
        """Fetch a snapshot of the order book."""

        url = (
            f"{self._BASE_URL}/order_book.snapshots.v1/exchanges/{exchange}/spot/{pair}/books/latest"
        )
        headers = {"X-Api-Key": self.api_key}
        params: dict[str, Any] = {"depth": depth}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
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

    async def _paginate(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
        max_pages: int | None = None,
    ) -> Iterable[dict]:
        """Internal helper handling Kaiko style pagination."""

        results: list[dict] = []
        page = 0
        async with httpx.AsyncClient() as client:
            next_page: str | None = None
            while True:
                req_params = params.copy()
                if next_page:
                    req_params["page_token"] = next_page
                async with self._sem:
                    resp = await client.get(url, params=req_params, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                results.extend(payload.get("data", []))
                next_page = payload.get("next_page_token")
                page += 1
                if not next_page or (max_pages and page >= max_pages):
                    break
        return results

    async def fetch_funding(
        self,
        exchange: str,
        pair: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        page_size: int = 1000,
        max_pages: int | None = None,
    ) -> List[Funding]:
        """Fetch historical funding rates for *pair* on *exchange*.

        Parameters are thin wrappers over Kaiko's REST API. Pagination is
        handled transparently and results are returned as a list of
        :class:`Funding` objects.
        """

        url = f"{self._BASE_URL}/funding-rates.v1/exchanges/{exchange}/perpetual/{pair}/historical"
        headers = {"X-Api-Key": self.api_key}
        params: dict[str, Any] = {"page_size": page_size}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        raw = await self._paginate(url, params, headers, max_pages)
        records: list[Funding] = []
        for f in raw:
            ts = f.get("timestamp") or f.get("time") or ""
            dt = (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, str) and ts
                else datetime.utcnow()
            )
            records.append(
                Funding(
                    timestamp=dt,
                    exchange=exchange,
                    symbol=pair,
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
        exchange: str,
        pair: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        page_size: int = 1000,
        max_pages: int | None = None,
    ) -> List[OpenInterest]:
        """Fetch historical open interest data for *pair*.

        The method automatically pages through the Kaiko API and normalises
        responses into :class:`OpenInterest` objects.
        """

        url = f"{self._BASE_URL}/open-interest.v1/exchanges/{exchange}/perpetual/{pair}/historical"
        headers = {"X-Api-Key": self.api_key}
        params: dict[str, Any] = {"page_size": page_size}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        raw = await self._paginate(url, params, headers, max_pages)
        records: list[OpenInterest] = []
        for r in raw:
            ts = r.get("timestamp") or r.get("time") or ""
            dt = (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, str) and ts
                else datetime.utcnow()
            )
            records.append(
                OpenInterest(
                    timestamp=dt,
                    exchange=exchange,
                    symbol=pair,
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

    # Backwards compatibility alias
    fetch_oi = fetch_open_interest
