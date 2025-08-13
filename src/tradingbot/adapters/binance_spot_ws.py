# src/tradingbot/adapters/binance_spot_ws.py
import asyncio
import json
import logging
import websockets
from datetime import datetime, timezone
from typing import AsyncIterator

from .base import ExchangeAdapter

log = logging.getLogger(__name__)

def _stream_name(symbol: str) -> str:
    return symbol.replace("/", "").lower() + "@trade"

class BinanceSpotWSAdapter(ExchangeAdapter):
    """
    WS de Binance Spot **TESTNET** para trades.
    """
    name = "binance_spot_testnet_ws"

    def __init__(self, ws_base: str | None = None):
        # Spot testnet: wss://testnet.binance.vision/stream?streams=
        self.ws_base = ws_base or "wss://testnet.binance.vision/stream?streams="

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        stream = _stream_name(symbol)
        url = self.ws_base + stream
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info("Conectado WS Spot testnet trades: %s", url)
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        d = msg.get("data") or {}
                        price = d.get("p")
                        qty = d.get("q")
                        ts_ms = d.get("T")
                        if price is None:
                            continue
                        price = float(price)
                        qty = float(qty or 0.0)
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
                        side = "sell" if d.get("m") else "buy"
                        yield {"ts": ts, "price": price, "qty": qty, "side": side}
            except Exception as e:
                log.warning("WS spot testnet desconectado (%s). Reintento en %.1fs ...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def place_order(self, *args, **kwargs) -> dict:
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError("no aplica en WS")
