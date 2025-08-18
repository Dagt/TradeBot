"""Basic execution algorithms returning order slices."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from .order_types import Order


def _clone(order: Order, qty: float) -> Order:
    return replace(order, qty=qty)


def twap(order: Order, slices: int) -> List[Order]:
    """Split ``order`` into ``slices`` with equal quantity."""
    if slices <= 0:
        return []
    qty = order.qty / slices
    return [_clone(order, qty) for _ in range(slices)]


def vwap(order: Order, volumes: Iterable[float]) -> List[Order]:
    """Distribute ``order`` quantity following ``volumes`` weights."""
    vols = list(volumes)
    total = sum(vols)
    if total <= 0:
        return []
    return [_clone(order, order.qty * v / total) for v in vols]


def pov(order: Order, trades: Iterable[dict], participation_rate: float) -> List[Order]:
    """Consume a percentage of traded volume until the order is filled."""
    executed = 0.0
    children: List[Order] = []
    for trade in trades:
        remaining = order.qty - executed
        if remaining <= 0:
            break
        qty = min(trade.get("qty", 0.0) * participation_rate, remaining)
        if qty <= 0:
            continue
        children.append(_clone(order, qty))
        executed += qty
    return children


__all__ = ["twap", "vwap", "pov"]
