import asyncio
from collections import defaultdict
from typing import Callable, Any

class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, cb: Callable[[Any], None]):
        self._subs[topic].append(cb)

    async def publish(self, topic: str, msg: Any):
        for cb in self._subs.get(topic, []):
            res = cb(msg)
            if asyncio.iscoroutine(res):
                await res
