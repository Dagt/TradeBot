import logging, asyncio
from ..bus import EventBus
from ..types import Tick

log = logging.getLogger(__name__)

async def run_trades_stream(adapter, symbol: str, bus: EventBus):
    async for d in adapter.stream_trades(symbol):
        # Normalizar (placeholder)
        tick = Tick(ts=d.get("ts"), exchange=adapter.name, symbol=symbol, price=d.get("price") or 0.0, qty=d.get("qty") or 0.0, side=d.get("side"))
        await bus.publish("trades", tick)
