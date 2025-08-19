"""Helpers for position sizing based on volatility targets."""

from __future__ import annotations


def vol_target(atr: float, risk_budget: float, notional_cap: float) -> float:
    """Return target position size for a given volatility estimate.

    Parameters
    ----------
    atr:
        Average true range or volatility estimate of the asset.
    risk_budget:
        Maximum notional exposure allowed based on risk appetite.
    notional_cap:
        Absolute cap on position size.

    Returns
    -------
    float
        The desired position size capped by ``notional_cap``. If any argument
        is non-positive, ``0.0`` is returned.
    """
    if atr <= 0 or risk_budget <= 0 or notional_cap <= 0:
        return 0.0

    size = risk_budget / atr
    return min(size, notional_cap)
