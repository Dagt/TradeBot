"""Utility helpers for fractional position sizing.

These helpers illustrate how ``Signal.strength`` maps to actual position
sizes under the fractional allocation model.
"""

from .base import Signal


def target_signal(side: str, current_fraction: float, target_fraction: float) -> Signal:
    """Return a :class:`Signal` aiming for ``target_fraction`` of the allocation.

    ``strength`` is set to ``target_fraction``.  When ``target_fraction`` is
    below ``current_fraction`` a ``reduce_only`` flag is emitted to ensure a
    partial exit.

    Examples
    --------
    >>> target_signal("buy", 0.3, 0.5)
    Signal(side='buy', strength=0.5, reduce_only=False)
    >>> target_signal("buy", 0.6, 0.2)
    Signal(side='buy', strength=0.2, reduce_only=True)
    """
    reduce = target_fraction < current_fraction
    return Signal(side, strength=target_fraction, reduce_only=reduce)


def strength_to_delta(max_position: float, current_qty: float, side: str, strength: float) -> float:
    """Translate ``strength`` into the position delta required.

    Parameters
    ----------
    max_position:
        Maximum absolute position permitted.
    current_qty:
        Current position quantity.
    side:
        ``"buy"`` or ``"sell"``.
    strength:
        Desired allocation fraction between 0 and 1.

    Returns
    -------
    float
        Signed quantity delta needed to reach the target allocation.

    Examples
    --------
    >>> strength_to_delta(100, 30, 'buy', 0.5)
    20.0
    >>> strength_to_delta(100, 30, 'sell', 0.2)
    -50.0
    """
    target = max_position if side == "buy" else -max_position if side == "sell" else 0.0
    desired = target * strength
    return desired - current_qty


__all__ = ["target_signal", "strength_to_delta"]
