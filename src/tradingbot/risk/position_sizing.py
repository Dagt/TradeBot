"""Helpers for position sizing based on volatility targets.

This module also provides utilities to translate signal strength into actual
position deltas via ``notional = equity * strength``, ensuring consistent
pyramiding and reduction across strategies. Local stop‑losses use ``risk_pct``
as ``notional * risk_pct``.
"""

from __future__ import annotations


def vol_target(atr: float, vol_target: float, equity: float) -> float:
    """Return target position size given a volatility estimate.

    Parameters
    ----------
    atr:
        Average true range or volatility estimate of the asset.
    vol_target:
        Fraction of current equity to allocate based on the desired
        volatility target.
    equity:
        Current account equity.

    Returns
    -------
    float
        Desired absolute position size.  If any argument is non-positive,
        ``0.0`` is returned.
    """
    if atr <= 0 or vol_target <= 0 or equity <= 0:
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

    ``strength`` controls notional through ``notional = equity * strength``.
    Values above ``1.0`` pyramid exposure, values between ``0`` and ``1`` scale
    it down and negatives close or flip the position. A separate ``risk_pct``
    can later apply a local stop‑loss as ``notional * risk_pct``.

    Parameters
    ----------
    strength:
        Target exposure as a fraction of account equity. Positive values denote
        long positions, negative values short. Values are clipped to
        ``[-1, 1]``.
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
    >>> delta_from_strength(-0.0, 10_000, 100, -40)
    40.0
    """

    strength = max(-1.0, min(1.0, strength))
    if price <= 0:
        return -current_qty
    target_qty = (equity * strength) / price
    return target_qty - current_qty

