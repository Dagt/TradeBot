"""Time Weighted Average Price execution algorithm."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from ..order_types import Order
from ..router import ExecutionRouter


@dataclass
class TWAP:
    """Split an order into equal slices executed over time."""

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
