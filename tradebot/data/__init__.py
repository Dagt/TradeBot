"""Data ingestion layer."""
from dataclasses import dataclass
from typing import Any

import ccxt.async_support as ccxt  # type: ignore


@dataclass
class ExchangeAdapter:
    """Adapter for interacting with exchange REST APIs using CCXT."""

    exchange: ccxt.Exchange

    async def fetch_ticker(self, symbol: str) -> Any:
        """Fetch ticker data for a symbol."""
        return await self.exchange.fetch_ticker(symbol)
