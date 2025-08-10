"""Risk management tools."""
from dataclasses import dataclass


@dataclass
class RiskManager:
    """Basic risk checks."""

    max_position: float = 1.0

    def check_position(self, size: float) -> bool:
        """Validate that proposed position size is within limits."""
        return abs(size) <= self.max_position
