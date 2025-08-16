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
        super().__init__()
        # Mainnet: wss://stream.binance.com:9443/stream?streams=
        # (Si usas testnet de futures, configura desde settings; lo dejamos parametrizable)
        self.ws_base = ws_base or "wss://stream.binance.com:9443/stream?streams="

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        stream = _binance_symbol_stream(self.normalize_symbol(symbol))
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
                        if price is not None:
                            self.state.last_px[symbol] = price
                        yield self.normalize_trade(symbol, ts, price, qty, side)
            except Exception as e:
                WS_FAILURES.labels(adapter=self.name).inc()
                log.warning("WS desconectado (%s). Reintentando en %.1fs ...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stream_order_book(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        stream = _binance_symbol_stream(self.normalize_symbol(symbol)).replace("@trade", f"@depth{depth}@100ms")
        url = self.ws_base + stream
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info("Conectado WS Binance orderbook: %s", url)
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        d = msg.get("data") or msg
                        bids = d.get("bids") or d.get("b") or []
                        asks = d.get("asks") or d.get("a") or []
                        bids_n = [[float(b[0]), float(b[1])] for b in bids]
                        asks_n = [[float(a[0]), float(a[1])] for a in asks]
                        ts_ms = d.get("E") or d.get("T")
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
                        self.state.order_book[symbol] = {"bids": bids_n, "asks": asks_n}
                        yield self.normalize_order_book(symbol, ts, bids_n, asks_n)
            except Exception as e:
                WS_FAILURES.labels(adapter=self.name).inc()
                log.warning("WS orderbook desconectado (%s). Reintento en %.1fs ...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    stream_orderbook = stream_order_book

    async def fetch_funding(self, symbol: str):
        raise NotImplementedError("WS adapter no soporta fetch_funding")

    async def fetch_oi(self, symbol: str):
        raise NotImplementedError("WS adapter no soporta fetch_oi")

    async def place_order(self, *args, **kwargs) -> dict:
        raise NotImplementedError("BinanceWSAdapter no gestiona órdenes")

    async def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError("BinanceWSAdapter no gestiona órdenes")
