import asyncio, logging
from typing import AsyncIterator
try:
    import ccxt
except Exception:  # pragma: no cover
    ccxt = None

from .base import ExchangeAdapter

log = logging.getLogger(__name__)

class BinanceAdapter(ExchangeAdapter):
    name = "binance"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        # Nota: para WebSocket L2/Trades se recomienda cliente WS nativo o ccxt.pro (comercial).
        self.rest = None
        if ccxt:
            self.rest = ccxt.binanceusdm() if testnet else ccxt.binance()
            if api_key and api_secret and self.rest:
                self.rest.apiKey = api_key
                self.rest.secret = api_secret
            if testnet and self.rest:
                # Futuros testnet
                self.rest.set_sandbox_mode(True)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        # Placeholder: en producción, usar websockets nativos para baja latencia.
        while True:
            await asyncio.sleep(1.0)
            yield {"ts": None, "price": None, "qty": None, "side": None}

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, price: float | None = None) -> dict:
        if not self.rest:
            raise RuntimeError("ccxt no disponible")
        params = {}
        if type_.lower() == "limit":
            resp = self.rest.create_order(symbol, "limit", side, qty, price, params)
        else:
            resp = self.rest.create_order(symbol, "market", side, qty, None, params)
        log.info("Order placed: %s", resp)
        return resp

    async def cancel_order(self, order_id: str) -> dict:
        # Implementar cancelación si es necesario (se requiere symbol para ccxt)
        return {"status": "not_implemented", "order_id": order_id}
