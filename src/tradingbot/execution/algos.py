"""Execution algorithms that split large orders into smaller slices.

These are simplified implementations intended for testing and examples.
Each algorithm uses an :class:`ExecutionRouter` to route the generated
sub-orders to the underlying adapter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List

from .order_types import Order
from .router import ExecutionRouter


@dataclass
class TWAP:
    """Time Weighted Average Price execution.

    Parameters
    ----------
    router: ExecutionRouter
        Router used to actually send sub-orders.
    slices: int
        Number of slices in which to split the parent order.
    delay: float, optional
        Delay in seconds between slices. Default ``0``.
    """

    router: ExecutionRouter
    slices: int
    delay: float = 0.0

    async def execute(self, order: Order) -> List[dict]:
        qty_per = order.qty / self.slices
        results: List[dict] = []
        for _ in range(self.slices):
            child = Order(
                symbol=order.symbol,
                side=order.side,
                type_=order.type_,
                qty=qty_per,
                price=order.price,
                post_only=order.post_only,
                time_in_force=order.time_in_force,
                iceberg_qty=order.iceberg_qty,
            )
            res = await self.router.execute(child)
            results.append(res)
            if self.delay:
                await asyncio.sleep(self.delay)
        return results


@dataclass
class VWAP:
    """Volume Weighted Average Price execution.

    The order is split according to the provided volume profile.
    """

    router: ExecutionRouter
    volumes: Iterable[float]
    delay: float = 0.0

    async def execute(self, order: Order) -> List[dict]:
        vols = list(self.volumes)
        total = sum(vols)
        results: List[dict] = []
        for v in vols:
            qty = order.qty * (v / total) if total else 0.0
            child = Order(
                symbol=order.symbol,
                side=order.side,
                type_=order.type_,
                qty=qty,
                price=order.price,
                post_only=order.post_only,
                time_in_force=order.time_in_force,
                iceberg_qty=order.iceberg_qty,
            )
            res = await self.router.execute(child)
            results.append(res)
            if self.delay:
                await asyncio.sleep(self.delay)
        return results


@dataclass
class POV:
    """Percentage of Volume execution.

    Consumes a percentage of the observed market volume.
    """

    router: ExecutionRouter
    participation_rate: float  # 0 < rate <= 1

    async def execute(self, order: Order, trades: AsyncIterator[dict]) -> List[dict]:
        executed = 0.0
        results: List[dict] = []
        async for trade in trades:
            remaining = order.qty - executed
            if remaining <= 0:
                break
            slice_qty = min(trade.get("qty", 0.0) * self.participation_rate, remaining)
            if slice_qty <= 0:
                continue
            child = Order(
                symbol=order.symbol,
                side=order.side,
                type_=order.type_,
                qty=slice_qty,
                price=order.price,
                post_only=order.post_only,
                time_in_force=order.time_in_force,
                iceberg_qty=order.iceberg_qty,
            )
            res = await self.router.execute(child)
            results.append(res)
            executed += slice_qty
        return results
