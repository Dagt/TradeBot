"""Helpers for position sizing based on volatility targets.

This module now also provides utilities to translate signal strength into
actual position deltas, ensuring consistent pyramiding and reduction across
strategies.
"""

from __future__ import annotations


def vol_target(atr: float, equity_pct: float, equity: float) -> float:
    """Return target position size given a volatility estimate.

    Parameters
    ----------
    atr:
        Average true range or volatility estimate of the asset.
    equity_pct:
        Fraction of current equity to allocate.
    equity:
        Current account equity.

    Returns
    -------
    float
        Desired absolute position size.  If any argument is non-positive,
        ``0.0`` is returned.
    """
    if atr <= 0 or equity_pct <= 0 or equity <= 0:
        return 0.0

    budget = equity * equity_pct
    return budget / atr


def delta_from_strength(
    strength: float,
    equity_pct: float,
    equity: float,
    price: float,
    current_qty: float,
) -> float:
    """Translate a signal ``strength`` into a position delta.

    Parameters
    ----------
    strength:
        Target exposure as a fraction of ``equity_pct``. Positive values denote
        long positions, negative values short. Values are clipped to
        ``[-1, 1]``.
    equity_pct:
        Fraction of current equity allocated to the asset.
    equity:
        Current account equity.
    price:
        Asset price used to convert notional into quantity.
    current_qty:
        Existing position size.

    Returns
    -------
    float
        Signed quantity required to reach the target exposure from
        ``current_qty``.

    Examples
    --------
    >>> delta_from_strength(0.5, 0.1, 10_000, 100, 20)
    30.0
    >>> delta_from_strength(0.2, 0.1, 10_000, 100, 30)
    -10.0
    >>> delta_from_strength(-0.0, 0.1, 10_000, 100, -40)
    40.0
    """

    strength = max(-1.0, min(1.0, strength))
    if price <= 0:
        return -current_qty
    target_qty = (equity * equity_pct * strength) / price
    return target_qty - current_qty

