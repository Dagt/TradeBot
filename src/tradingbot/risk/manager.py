"""Equity based risk manager.

This module implements a simplified risk manager that sizes positions based on
account equity instead of absolute position limits.  It replaces the previous
``RiskManager`` which worked with a fixed ``max_pos`` and volatility based
adjustments.

The new :class:`EquityRiskManager` receives a ``risk_pct`` (e.g. ``0.01`` for
1%) and a callable ``equity_provider`` which should return the current account
equity in quote currency.  The manager calculates the amount of capital that
can be put at risk for a trade and derives the purchasable quantity given an
asset price.

If the available equity does not cover at least the price of a single unit the
manager will return ``0`` to signal that no trade should be placed.

The implementation intentionally keeps a very small surface area: position
tracking, basic kill switch handling and aggregation helpers used by other
modules.  More advanced features from the previous version (volatility sizing,
drawdown checks, correlation limits, etc.) have been removed in favour of the
equity based approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict


@dataclass
class Position:
    """Container for the current position size."""

    qty: float = 0.0


class EquityRiskManager:
    """Simple equity based risk manager."""

    def __init__(
        self,
        risk_pct: float,
        equity_provider: Callable[[], float],
    ) -> None:
        self.risk_pct = abs(float(risk_pct))
        self.equity_provider = equity_provider

        self.pos = Position()
        self.enabled = True
        self.last_kill_reason: str | None = None

        # Multi-exchange position book: {exchange: {symbol: qty}}
        self.positions_multi: Dict[str, Dict[str, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Position helpers
    def set_position(self, qty: float) -> None:
        self.pos.qty = float(qty)

    def add_fill(self, side: str, qty: float) -> None:
        signed = float(qty) if side == "buy" else -float(qty)
        self.pos.qty += signed

    # ------------------------------------------------------------------
    def _risk_capital(self) -> float:
        equity = float(self.equity_provider() or 0.0)
        return equity * self.risk_pct

    def size(self, signal_side: str, price: float, strength: float = 1.0) -> float:
        """Return desired position delta based on equity percentage.

        Parameters
        ----------
        signal_side:
            ``"buy"`` to target a long position or ``"sell"`` for short.
        price:
            Current price of the asset.
        strength:
            Optional multiplier to scale the desired target.
        """

        if price <= 0 or not self.enabled:
            return 0.0

        capital = self._risk_capital()
        if capital < price:
            return 0.0

        max_qty = capital / price
        target = max_qty if signal_side == "buy" else -max_qty if signal_side == "sell" else 0.0
        delta = (target - self.pos.qty) * strength
        return delta

    # ------------------------------------------------------------------
    # Kill switch helpers
    def kill_switch(self, reason: str) -> None:
        self.enabled = False
        self.last_kill_reason = reason

    def reset(self) -> None:
        self.enabled = True
        self.last_kill_reason = None
        self.pos.qty = 0.0

    # ------------------------------------------------------------------
    # Multi-exchange position helpers
    def update_position(self, exchange: str, symbol: str, qty: float) -> None:
        book = self.positions_multi.setdefault(exchange, {})
        book[symbol] = float(qty)

    def aggregate_positions(self) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        for book in self.positions_multi.values():
            for sym, qty in book.items():
                agg[sym] = agg.get(sym, 0.0) + float(qty)
        return agg

    def close_all_positions(self) -> None:
        self.pos.qty = 0.0
        for exch in list(self.positions_multi.keys()):
            for sym in list(self.positions_multi[exch].keys()):
                self.positions_multi[exch][sym] = 0.0


# Backwards compatibility alias
RiskManager = EquityRiskManager

