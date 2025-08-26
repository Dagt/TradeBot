from __future__ import annotations
from typing import Mapping, Protocol


class EquityProvider(Protocol):
    """Interface for objects exposing equity and cash information."""

    def equity(self, mark_prices: Mapping[str, float] | None = None) -> float:
        """Return total account equity in quote currency."""
        ...

    def available_cash(self) -> float:
        """Return free quote currency available for new positions."""
        ...


__all__ = ["EquityProvider"]
