"""Trading strategies."""
from dataclasses import dataclass
from typing import Any


@dataclass
class Strategy:
    """Base strategy class."""

    name: str

    def generate_signal(self, data: Any) -> float:
        """Return a trading signal given market data."""
        raise NotImplementedError
