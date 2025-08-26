"""Helpers for position sizing based on volatility targets.

This module now also provides utilities to translate signal strength into
actual position deltas, ensuring consistent pyramiding and reduction across
strategies.
"""

from __future__ import annotations


def vol_target(atr: float, equity: float, vol_target: float) -> float:
    """Return target position size given a volatility estimate.

    Parameters
    ----------
    atr:
        Average true range or volatility estimate of the asset.
    equity:
        Current account equity.
    vol_target:
        Fraction of equity to allocate based on the volatility target.

    Returns
    -------
    float
        Desired absolute position size.  If any argument is non-positive,
        ``0.0`` is returned.

    Examples
    --------
    >>> vol_target(atr=2.0, equity=10.0, vol_target=1.0)
    5.0
    """
    if atr <= 0 or equity <= 0 or vol_target <= 0:
        return 0.0

    budget = equity * vol_target
    return budget / atr


def delta_from_strength(
    strength: float,
    equity: float,
    price: float,
    current_qty: float,
) -> float:
    """Translate a signal ``strength`` into a position delta.

    Parameters
    ----------
    strength:
        Target exposure as a fraction of total equity. Positive values denote
        long positions, negative values short.
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
    >>> delta_from_strength(0.5, 10_000, 100, 20)
    30.0
    >>> delta_from_strength(0.2, 10_000, 100, 30)
    -10.0
    >>> delta_from_strength(0.0, 10_000, 100, -40)
    40.0
    """

    if price <= 0:
        return -current_qty
    target_qty = (equity * strength) / price
    return target_qty - current_qty

