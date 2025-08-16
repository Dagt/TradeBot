from __future__ import annotations

from typing import AsyncIterator

from .base import ExchangeAdapter
from .deribit import DeribitAdapter


class DeribitWSAdapter(ExchangeAdapter):
    """Websocket wrapper delegando a :class:`DeribitAdapter` para REST."""

    name = "deribit_ws"

    def __init__(self, rest: DeribitAdapter | None = None):
        super().__init__()
        self.rest = rest

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        # Implementación WS real omitida; no usamos streaming en tests
        if False:
            yield {}

    async def fetch_funding(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_funding(symbol)
        raise NotImplementedError("Funding no disponible")

    async def fetch_basis(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_basis(symbol)
        raise NotImplementedError("Basis no disponible")

    async def fetch_oi(self, symbol: str):
        if self.rest:
            return await self.rest.fetch_oi(symbol)
        raise NotImplementedError("Open interest no disponible")

    fetch_open_interest = fetch_oi

    async def place_order(self, *args, **kwargs) -> dict:
        if self.rest:
            return await self.rest.place_order(*args, **kwargs)
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str) -> dict:
        if self.rest:
            return await self.rest.cancel_order(order_id)
        raise NotImplementedError("no aplica en WS")
