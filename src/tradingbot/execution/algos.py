from __future__ import annotations

"""Simple execution algorithms that produce order slices."""

from typing import Iterable, List, Tuple

from .order_types import Order


OrderSlice = Tuple[Order, float]


def _clone(order: Order, qty: float) -> Order:
    return Order(
        symbol=order.symbol,
        side=order.side,
        type_=order.type_,
        qty=qty,
        price=order.price,
        post_only=order.post_only,
        time_in_force=order.time_in_force,
        iceberg_qty=order.iceberg_qty,
    )


def twap(order: Order, slices: int, interval: float = 0.0) -> List[OrderSlice]:
    """Time-weighted average price algorithm.

    Splits ``order`` into ``slices`` equally sized child orders separated by
    ``interval`` seconds.
    """

    qty = order.qty / slices
    children: List[OrderSlice] = []
    for i in range(slices):
        child = _clone(order, qty)
        delay = 0.0 if i == 0 else interval
        children.append((child, delay))
    return children


def vwap(
    order: Order, volumes: Iterable[float], interval: float = 0.0
) -> List[OrderSlice]:
    """Volume-weighted average price algorithm.

    ``volumes`` is a sequence of relative market volumes to match against. Each
    element determines the proportion of ``order.qty`` to send in that slice.
    Child orders are separated by ``interval`` seconds.
    """

    weights = list(volumes)
    total = sum(weights)
    children: List[OrderSlice] = []
    for i, w in enumerate(weights):
        qty = order.qty * (w / total) if total else 0.0
        child = _clone(order, qty)
        delay = 0.0 if i == 0 else interval
        children.append((child, delay))
    return children


def pov(
    order: Order, trades: Iterable[dict], participation_rate: float
) -> List[OrderSlice]:
    """Participation of volume algorithm.

    ``trades`` is an iterable of dictionaries containing ``qty`` and ``ts``
    (timestamp in seconds). For each trade we send an order with quantity equal
    to ``participation_rate`` times the trade quantity, capped by the remaining
    order quantity. Delays between slices are derived from consecutive trade
    timestamps. Any residual quantity is sent immediately.
    """

    remaining = order.qty
    children: List[OrderSlice] = []
    prev_ts: float | None = None
    for tr in trades:
        if remaining <= 0:
            break
        trade_qty = tr.get("qty", 0.0)
        slice_qty = min(remaining, trade_qty * participation_rate)
        if slice_qty <= 0:
            prev_ts = tr.get("ts")
            continue
        child = _clone(order, slice_qty)
        ts = tr.get("ts")
        delay = 0.0 if prev_ts is None else max(0.0, float(ts) - float(prev_ts))
        children.append((child, delay))
        remaining -= slice_qty
        prev_ts = ts
    if remaining > 0:
        children.append((_clone(order, remaining), 0.0))
    return children


__all__ = ["twap", "vwap", "pov", "OrderSlice"]
