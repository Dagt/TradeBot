"""Percentage of Volume execution algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, List

from ..order_types import Order
from ..router import ExecutionRouter


@dataclass
class POV:
    """Consume a percentage of observed market volume."""

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
