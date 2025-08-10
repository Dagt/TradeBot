from abc import ABC, abstractmethod
from typing import AsyncIterator

class ExchangeAdapter(ABC):
    name: str

    @abstractmethod
    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        """Yield dicts with ts, price, qty, side."""

    @abstractmethod
    async def place_order(self, symbol: str, side: str, type_: str, qty: float, price: float | None = None) -> dict:
        """Return provider response (paper/live)."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> dict:
        ...
