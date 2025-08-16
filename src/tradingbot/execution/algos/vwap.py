"""Volume Weighted Average Price execution algorithm."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List

from ..order_types import Order
from ..router import ExecutionRouter


@dataclass
class VWAP:
    """Distribute an order following a volume profile."""

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
