import logging, asyncio
from ..bus import EventBus
from ..types import Tick, OrderBook
from ..storage.timescale import insert_orderbook

log = logging.getLogger(__name__)

async def run_trades_stream(adapter, symbol: str, bus: EventBus):
    async for d in adapter.stream_trades(symbol):
        # Normalizar (placeholder)
        tick = Tick(
            ts=d.get("ts"),
            exchange=adapter.name,
            symbol=symbol,
            price=d.get("price") or 0.0,
            qty=d.get("qty") or 0.0,
            side=d.get("side"),
        )
        await bus.publish("trades", tick)


async def run_orderbook_stream(adapter, symbol: str, depth: int, bus: EventBus, engine):
    async for d in adapter.stream_orderbook(symbol, depth):
        ob = OrderBook(
            ts=d.get("ts"),
            exchange=adapter.name,
            symbol=symbol,
            bid_px=d.get("bid_px") or [],
            bid_qty=d.get("bid_qty") or [],
            ask_px=d.get("ask_px") or [],
            ask_qty=d.get("ask_qty") or [],
        )
        await bus.publish("orderbook", ob)
        insert_orderbook(
            engine,
            ts=ob.ts,
            exchange=ob.exchange,
            symbol=ob.symbol,
            bid_px=ob.bid_px,
            bid_qty=ob.bid_qty,
            ask_px=ob.ask_px,
            ask_qty=ob.ask_qty,
        )
