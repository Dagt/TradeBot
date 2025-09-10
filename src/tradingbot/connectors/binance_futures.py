"""Connector for Binance USD-M futures using CCXT."""
from __future__ import annotations

from .base import ExchangeConnector


class BinanceFuturesConnector(ExchangeConnector):
    """Minimal connector for Binance futures markets."""

    # CCXT client uses the ``binanceusdm`` identifier for USD-M futures
    name = "binanceusdm"

    async def fetch_balance(self) -> dict[str, float]:
        """Return account balances with numeric totals as floats."""

        data = await self._rest_call(self.rest.fetch_balance)
        balances: dict[str, float] = {}
        for asset, info in (data or {}).items():
            if isinstance(info, dict) and "total" in info:
                try:
                    balances[asset] = float(info["total"])
                except (TypeError, ValueError):
                    continue
        return balances

    async def fetch_ticker(self, symbol: str) -> float | None:
        """Return the last traded price for ``symbol``.

        The response from CCXT varies slightly between exchanges; the helper
        normalises the common fields and returns ``None`` when no price is
        available.
        """

        data = await self._rest_call(self.rest.fetch_ticker, symbol)
        price = (
            data.get("last")
            or data.get("close")
            or data.get("price")
            or data.get("bid")
            or data.get("ask")
        )
        return float(price) if price is not None else None
