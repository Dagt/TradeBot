"""Utility functions for estimating slippage and queue position.

These helpers are used by :mod:`tradingbot.execution.router` to estimate the
potential cost of an order before it is sent to an exchange.  The estimates are
based solely on the visible order book and therefore provide a best effort
approximation only.
"""
from __future__ import annotations

from typing import Iterable, Tuple


def impact_by_depth(side: str, qty: float, levels: Iterable[Tuple[float, float]]) -> float | None:
    """Estimate slippage in basis points from consuming order book depth.

    Parameters
    ----------
    side:
        Order side, ``"buy"`` or ``"sell"``.
    qty:
        Quantity to execute.
    levels:
        Iterable of ``(price, quantity)`` tuples sorted from best to worst
        price for the corresponding side of the book.

    Returns
    -------
    float | None
        Estimated slippage in basis points relative to the top of book price.
        ``None`` is returned if the available depth is insufficient to fully
        execute the order.
    """
    levels = list(levels)
    if not levels or qty <= 0:
        return None

    top_px, _ = levels[0]
    remaining = qty
    cost = 0.0
    for px, level_qty in levels:
        take = min(remaining, level_qty)
        cost += px * take
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:  # not enough depth
        return None

    avg_px = cost / qty
    direction = 1 if side.lower() == "buy" else -1
    return (avg_px - top_px) / top_px * 10000 * direction


def queue_position(qty: float, levels: Iterable[Tuple[float, float]]) -> float:
    """Estimate queue position when joining the best price level.

    Parameters
    ----------
    qty:
        Quantity of our order.
    levels:
        Iterable of ``(price, quantity)`` tuples representing the resting
        orders at the relevant book side.  Only the first level is used.

    Returns
    -------
    float
        Fraction representing the expected position in queue.  ``0.0`` means no
        resting volume ahead, ``1.0`` means we are entirely behind existing
        volume.
    """
    levels = list(levels)
    if not levels:
        return 0.0
    _, top_qty = levels[0]
    if top_qty <= 0:
        return 0.0
    return top_qty / (top_qty + qty)


__all__ = ["impact_by_depth", "queue_position"]
