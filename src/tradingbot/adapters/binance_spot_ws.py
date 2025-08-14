# src/tradingbot/adapters/binance_spot_ws.py
import asyncio
import json
import logging
import websockets
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

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
                        yield {"symbol": symbol, "ts": ts, "price": price, "qty": qty, "side": side}
            except Exception as e:
                log.warning("WS spot testnet desconectado (%s). Reintento en %.1fs ...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stream_trades_multi(self, symbols: Iterable[str]) -> AsyncIterator[dict]:
        """
        Un solo socket con múltiples streams. Yields:
        {"symbol": <sym>, "ts": datetime, "price": float, "qty": float, "side": "buy"/"sell"}
        """
        streams = "/".join(_stream_name(s) for s in symbols)
        url = self.ws_base + streams
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    log.info("Conectado WS Spot testnet multi: %s", url)
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        stream = (msg.get("stream") or "")
                        data = msg.get("data") or {}
                        price = data.get("p")
                        qty = data.get("q")
                        ts_ms = data.get("T")
                        if price is None:
                            continue
                        symbol_key = stream.split("@")[0].upper()
                        # re‑insertamos la barra (BTCUSDT -> BTC/USDT)
                        if symbol_key.endswith("USDT"):
                            base = symbol_key[:-4]
                            symbol = f"{base}/USDT"
                        else:
                            # fallback simple (si hay otros quotes)
                            # intenta separar último 3/4 chars como quote comunes
                            candidates = ("USDT","BUSD","BTC","ETH","BNB","FDUSD","TUSD")
                            match = next((q for q in candidates if symbol_key.endswith(q)), None)
                            symbol = f"{symbol_key[:-len(match)]}/{match}" if match else symbol_key

                        price = float(price)
                        qty = float(qty or 0.0)
                        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
                        side = "sell" if data.get("m") else "buy"
                        yield {"symbol": symbol, "ts": ts, "price": price, "qty": qty, "side": side}
            except Exception as e:
                log.warning("WS spot testnet multi desconectado (%s). Reintento en %.1fs ...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def place_order(self, *args, **kwargs) -> dict:
        raise NotImplementedError("solo streaming")

    async def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError("no aplica en WS")
