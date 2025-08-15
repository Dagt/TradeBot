# src/tradingbot/adapters/binance_ws.py
import asyncio
import json
import logging
import websockets
from datetime import datetime, timezone
from typing import AsyncIterator

from .base import ExchangeAdapter
from ..utils.metrics import WS_FAILURES

log = logging.getLogger(__name__)

def _binance_symbol_stream(symbol: str) -> str:
    # Binance usa minúsculas y sin separador para streams (BTCUSDT -> btcusdt)
    return symbol.replace("/", "").lower() + "@trade"

class BinanceWSAdapter(ExchangeAdapter):
    """
    Solo streaming de trades vía WS (no órdenes). Útil para alimentar precios al paper broker.
    """
    name = "binance"

    def __init__(self, ws_base: str | None = None):
        # Mainnet: wss://stream.binance.com:9443/stream?streams=
        # (Si usas testnet de futures, configura desde settings; lo dejamos parametrizable)
        self.ws_base = ws_base or "wss://stream.binance.com:9443/stream?streams="

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        stream = _binance_symbol_stream(symbol)
        url = self.ws_base + stream
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info("Conectado WS Binance trades: %s", url)
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        d = msg.get("data") or {}
                        # Trade payload (t=trade id, p=price, q=qty, T=trade time ms, m=is buyer maker)
                        price = float(d.get("p")) if d.get("p") is not None else None
                        qty = float(d.get("q")) if d.get("q") is not None else None
                        ts_ms = d.get("T")
                        side = "sell" if d.get("m") else "buy"  # buyer is maker -> aggressive sell
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
                        yield {"ts": ts, "price": price, "qty": qty, "side": side}
            except Exception as e:
                WS_FAILURES.labels(adapter=self.name).inc()
                log.warning("WS desconectado (%s). Reintentando en %.1fs ...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def place_order(self, *args, **kwargs) -> dict:
        raise NotImplementedError("BinanceWSAdapter no gestiona órdenes")

    async def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError("BinanceWSAdapter no gestiona órdenes")
