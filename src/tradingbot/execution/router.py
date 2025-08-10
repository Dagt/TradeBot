import logging
from .order_types import Order

log = logging.getLogger(__name__)

class ExecutionRouter:
    def __init__(self, adapter):
        self.adapter = adapter

    async def execute(self, order: Order) -> dict:
        log.info("Routing order %s", order)
        return await self.adapter.place_order(
            symbol=order.symbol,
            side=order.side,
            type_=order.type_,
            qty=order.qty,
            price=order.price
        )
