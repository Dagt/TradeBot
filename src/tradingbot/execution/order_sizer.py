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
) -> float:
    """Return ``qty`` rounded to ``step_size`` and validated against ``min_notional``.

    If the resulting notional falls below ``min_notional`` the function returns
    ``0`` signalling that the order should be skipped.
    """
    if price <= 0:
        return 0.0
    notional = qty * price
    if min_notional and notional < min_notional:
        return 0.0
    qty = _round_step(qty, step_size or 0.0)
    if min_notional and qty * price < min_notional:
        return 0.0
    return qty
