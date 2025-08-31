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
    open_orders: Dict[str, float] = field(default_factory=dict)

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
        """Return cash available after reserving capital for open orders."""
        reserved = sum(
            abs(qty) * float(self.prices.get(sym, 0.0))
            for sym, qty in self.open_orders.items()
        )
        return float(self.cash) - reserved

    def update_open_order(self, symbol: str, delta_qty: float) -> None:
        """Adjust pending quantity for ``symbol`` by ``delta_qty``."""
        qty = self.open_orders.get(symbol, 0.0) + float(delta_qty)
        if abs(qty) > 0.0:
            self.open_orders[symbol] = qty
        else:
            self.open_orders.pop(symbol, None)

    def pending_exposure(self, symbol: str) -> float:
        """Return notional exposure tied up in open orders for ``symbol``."""
        qty = abs(self.open_orders.get(symbol, 0.0))
        price = float(self.prices.get(symbol, 0.0))
        return qty * price

    def current_exposure(self, symbol: str) -> tuple[float, float]:
        """Return net quantity and notional exposure for ``symbol``.

        The returned notional is the absolute quantity multiplied by the last
        marked price for the symbol.  If the symbol has not been priced yet the
        notional defaults to ``0.0``.
        """
        qty = float(self.positions.get(symbol, 0.0))
        price = float(self.prices.get(symbol, 0.0))
        return qty, abs(qty) * price
