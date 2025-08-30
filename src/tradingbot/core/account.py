"""Basic account state tracking cash and positions.

This module introduces the :class:`Account` class which keeps a lightweight
record of cash balances and open positions for a trading venue.  It exposes a
couple of helper methods used across runners and adapters to query the available
cash and the current exposure for a given symbol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Account:
    """Container for balances and positions.

    Parameters
    ----------
    max_symbol_exposure:
        Maximum notional exposure allowed per symbol.  This limit is stored but
        enforcement is left to higher level components.
    cash:
        Optional starting cash balance.
    """

    max_symbol_exposure: float
    cash: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def update_cash(self, delta: float) -> None:
        """Adjust available cash by ``delta``."""
        self.cash += float(delta)

    def update_position(self, symbol: str, delta_qty: float, price: float | None = None) -> None:
        """Update net position for ``symbol`` and optionally its last price."""
        self.positions[symbol] = self.positions.get(symbol, 0.0) + float(delta_qty)
        if price is not None:
            self.prices[symbol] = float(price)

    def mark_price(self, symbol: str, price: float) -> None:
        """Record latest mark price for ``symbol``."""
        self.prices[symbol] = float(price)

    # ------------------------------------------------------------------
    def get_available_balance(self) -> float:
        """Return available cash excluding unrealised PnL."""
        return float(self.cash)

    def current_exposure(self, symbol: str) -> tuple[float, float]:
        """Return net quantity and notional exposure for ``symbol``.

        The returned notional is the absolute quantity multiplied by the last
        marked price for the symbol.  If the symbol has not been priced yet the
        notional defaults to ``0.0``.
        """
        qty = float(self.positions.get(symbol, 0.0))
        price = float(self.prices.get(symbol, 0.0))
        return qty, abs(qty) * price
