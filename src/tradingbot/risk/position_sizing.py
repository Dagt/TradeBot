"""Helpers for position sizing based on volatility targets.

This module now also provides utilities to translate signal strength into
actual position deltas, ensuring consistent pyramiding and reduction across
strategies.
"""

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


def delta_from_strength(
    side: str,
    strength: float,
    max_pos: float,
    current: float,
) -> float:
    """Convert a strength fraction into a position delta.

    Parameters
    ----------
    side:
        ``"buy"`` or ``"sell"`` indicating the desired direction.
    strength:
        Desired exposure as a fraction of ``max_pos``. Values are clipped to
        ``[0, 1]`` with ``0`` meaning the existing position should be closed.
    max_pos:
        Maximum absolute position size allowed.
    current:
        Current signed exposure.

    Returns
    -------
    float
        Signed quantity needed to move from ``current`` to the target
        exposure. Positive values indicate buys, negative sells.

    Examples
    --------
    >>> delta_from_strength("buy", 0.5, 100, 20)
    30.0
    >>> delta_from_strength("buy", 0.2, 100, 30)
    -10.0
    >>> delta_from_strength("sell", 0.0, 100, -40)
    40.0
    """

    strength = max(0.0, min(1.0, strength))
    if strength <= 0:
        return -current
    target = max_pos * strength
    target = target if side == "buy" else -target
    return target - current

