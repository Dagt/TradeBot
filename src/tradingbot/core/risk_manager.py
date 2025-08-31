"""Basic position and risk management helpers.

This module implements a small :class:`RiskManager` that interacts with the
:class:`~tradingbot.core.account.Account` class to size positions and manage
stops.  The functionality is intentionally lightweight – it is mostly meant for
examples and small trading experiments.  The manager keeps no market data of its
own besides what is supplied in method calls.

Main features
-------------
    * Size positions based on a fixed percentage of available balance and an input
      signal strength.
    * Calculate initial stop prices using a fixed percentage of the entry price.
    * Progressively tighten stops as a trade becomes profitable using four stages:
      1. cut the initial risk in half,
      2. move the stop to break‑even once price moves the full initial risk,
      3. lock in one dollar of net profit (after fees and slippage),
      4. afterwards trail the stop at ``2 × ATR``.
    * Check global exposure limits before submitting new orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

from .account import Account


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Retrieve ``name`` from ``obj`` whether it's a mapping or attribute."""
    if isinstance(obj, MutableMapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value: Any) -> None:
    """Set ``name`` on ``obj`` whether it's a mapping or attribute."""
    if isinstance(obj, MutableMapping):
        obj[name] = value
    else:
        setattr(obj, name, value)


@dataclass
class RiskManager:
    """Minimalistic risk manager.

    Parameters
    ----------
    account:
        Instance of :class:`Account` used to query balances and exposures.
    risk_per_trade:
        Fraction of available balance to risk per trade.  A value of ``0.1``
        corresponds to risking 10% of cash on each trade.  This value can be
        overridden per trade when calling :meth:`calc_position_size`.
    atr_mult:
        Multiplier applied to the ATR when calculating stops.
    """

    account: Account
    risk_per_trade: float = 0.1
    atr_mult: float = 2.0
    risk_pct: float = 0.01

    # ------------------------------------------------------------------
    # Position sizing
    def calc_position_size(
        self,
        signal_strength: float,
        price: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """Return position size given ``signal_strength`` and ``price``.

        ``signal_strength`` is expected to be in the ``[0, 1]`` range and
        linearly scales the capital fraction used.  A ``signal_strength`` of
        ``1`` risks the full ``risk_per_trade`` portion of available balance,
        while ``0.5`` uses half of it.  The method does **not** clamp the
        signal so callers can experiment with their own conventions.

        Parameters
        ----------
        signal_strength:
            Trade conviction in the ``[0, 1]`` range.
        price:
            Current asset price.
        risk_per_trade:
            Optional override for the fraction of balance to risk.  If ``None``
            (default) the manager's ``risk_per_trade`` attribute is used.
        """

        balance = float(self.account.get_available_balance())
        rpt = self.risk_per_trade if risk_per_trade is None else float(risk_per_trade)
        alloc = balance * rpt * float(signal_strength)
        if price <= 0:
            return 0.0
        size = alloc / float(price)
        return max(0.0, size)

    # ------------------------------------------------------------------
    # Stop management helpers
    def initial_stop(self, entry_price: float, side: str, atr: float | None = None) -> float:
        """Return the initial stop price for a new position."""

        delta = float(entry_price) * float(self.risk_pct)
        if side.lower() in {"buy", "long"}:
            return float(entry_price) - delta
        return float(entry_price) + delta

    def update_trailing(self, trade: Any, current_price: float, fees_slip: float = 0.0) -> None:
        """Update ``trade`` stop according to the four stop stages."""

        side = _get(trade, "side")
        entry = float(_get(trade, "entry_price"))
        stop = float(_get(trade, "stop"))
        atr = float(_get(trade, "atr", 0.0))
        stage = int(_get(trade, "stage", 0))
        qty = float(_get(trade, "qty", 1.0))

        # Distance from entry to stop defines the initial risk per unit
        risk = abs(entry - stop)
        move = current_price - entry if side in {"buy", "long"} else entry - current_price

        # Stage 0 -> 1: price moves half the risk, cut risk in half
        if stage == 0 and move >= risk / 2:
            new_stop = entry - risk / 2 if side in {"buy", "long"} else entry + risk / 2
            _set(trade, "stop", new_stop)
            _set(trade, "stage", 1)
            return

        # Stage 1 -> 2: move stop to break-even after full risk move
        if stage == 1 and move >= risk:
            _set(trade, "stop", entry)
            _set(trade, "stage", 2)
            return

        # Stage 2 -> 3: lock in +1 USD net after fees/slippage
        net = move * qty - float(fees_slip)
        if stage <= 2 and net >= 1.0:
            price_shift = (1.0 + float(fees_slip)) / qty
            new_stop = entry + price_shift if side in {"buy", "long"} else entry - price_shift
            _set(trade, "stop", new_stop)
            _set(trade, "stage", 3)
            return

        # Stage 3 -> 4+: trailing at 2 * ATR
        if stage >= 3 and atr > 0:
            trail = current_price - 2 * atr if side in {"buy", "long"} else current_price + 2 * atr
            if side in {"buy", "long"}:
                if trail > stop:
                    _set(trade, "stop", trail)
            else:
                if trail < stop:
                    _set(trade, "stop", trail)
            _set(trade, "stage", max(stage, 4))

    # ------------------------------------------------------------------
    def manage_position(self, trade: Any, signal: dict | None = None) -> str:
        """Decide whether to hold or close ``trade``.

        The ``trade`` object is expected to provide ``side``, ``stop`` and a
        ``current_price`` attribute (or corresponding mapping keys).  When
        ``signal`` specifies an opposite side or includes an ``exit`` flag the
        method returns ``"close"``; otherwise ``"hold"``.
        """

        side = _get(trade, "side")
        price = _get(trade, "current_price")
        stop = _get(trade, "stop")

        if price is not None and stop is not None:
            if side in {"buy", "long"} and float(price) <= float(stop):
                return "close"
            if side in {"sell", "short"} and float(price) >= float(stop):
                return "close"

        if signal:
            sig_side = signal.get("side")
            if sig_side and sig_side.lower() not in {side.lower()}:
                return "close"
            if signal.get("exit"):
                return "close"

            strength = signal.get("strength")
            cur_strength = float(_get(trade, "strength", 1.0))
            if sig_side and sig_side.lower() == side.lower() and strength is not None:
                if strength > cur_strength:
                    _set(trade, "strength", strength)
                    return "scale_in"
                if strength < cur_strength:
                    _set(trade, "strength", strength)
                    if strength <= 0:
                        return "close"
                    return "scale_out"
        return "hold"

    # ------------------------------------------------------------------
    def check_global_exposure(self, symbol: str, new_alloc: float) -> bool:
        """Return ``True`` if ``new_alloc`` keeps exposure within limits."""

        _, current = self.account.current_exposure(symbol)
        limit = float(self.account.max_symbol_exposure)
        return current + abs(float(new_alloc)) <= limit
