"""Persistence layer for market data and trades."""
from dataclasses import dataclass


@dataclass
class Database:
    """Placeholder database interface."""

    dsn: str = "postgresql://localhost:5432/tradebot"

    def connect(self) -> None:
        """Connect to the database (placeholder)."""
        pass
