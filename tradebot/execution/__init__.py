"""Order execution and routing."""
from dataclasses import dataclass
from typing import Any


@dataclass
class OrderRouter:
    """Simple order router stub."""

    async def submit_order(self, order: Any) -> None:
        """Submit an order to the exchange."""
        raise NotImplementedError
