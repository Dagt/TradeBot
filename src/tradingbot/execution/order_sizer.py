"""Common order sizing helpers for live and paper runners.

These utilities mimic the sizing behaviour of the backtester without altering
its code.  They allow the execution side to skip tiny orders that would be
rounded up by venue rules and align live behaviour with historical simulations.
"""
from __future__ import annotations
import math


def _round_step(qty: float, step: float) -> float:
    if not step:
        return qty
    return math.floor(qty / step) * step


def adjust_qty(
    qty: float,
    price: float,
    min_notional: float | None = None,
    step_size: float | None = None,
    min_qty: float | None = None,
) -> float:
    """Return ``qty`` rounded to venue constraints and validated.

    Orders smaller than ``min_qty`` or ``min_notional`` are rejected by
    returning ``0`` so that callers can skip placing them.
    """
    if price <= 0:
        return 0.0
    if min_qty and abs(qty) < min_qty:
        return 0.0
    notional = qty * price
    if min_notional and abs(notional) < min_notional:
        return 0.0
    qty = _round_step(qty, step_size or 0.0)
    if min_qty and abs(qty) < min_qty:
        return 0.0
    if min_notional and abs(qty * price) < min_notional:
        return 0.0
    return qty
