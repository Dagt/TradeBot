"""Backtesting utilities."""
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Backtester:
    """Very small backtester stub."""

    prices: Iterable[float]

    def run(self) -> float:
        """Run the backtest and return dummy PnL."""
        return sum(self.prices)
