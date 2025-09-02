import asyncio
import time
import contextlib

import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter


class HighFrequencyAdapter:
    """Adapter that processes orders through an async queue."""

    name = "hf"

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.processed = 0
        self.max_qsize = 0
        self.worker = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            self.processed += 1
            self.queue.task_done()

    async def place_order(self, **kwargs) -> dict:
        await self.queue.put(kwargs)
        self.max_qsize = max(self.max_qsize, self.queue.qsize())
        return {"status": "filled", "qty": kwargs.get("qty"), "price": kwargs.get("price", 1.0)}


@pytest.mark.asyncio
async def test_high_frequency_router():
    adapter = HighFrequencyAdapter()
    router = ExecutionRouter(adapter)
    messages = 2000
    tasks = []
    start = time.perf_counter()
    for _ in range(messages):
        order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=1.0)
        tasks.append(asyncio.create_task(router.execute(order)))
        # Yield control so the adapter can drain the queue.
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    await adapter.queue.join()

    assert adapter.processed == messages
    assert adapter.queue.qsize() == 0
    assert adapter.max_qsize < 5
    assert messages / duration > 1000

    adapter.worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await adapter.worker
