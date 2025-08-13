# src/tradingbot/adapters/binance_futures.py
from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator, Optional

try:
    import ccxt
except Exception:
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings

log = logging.getLogger(__name__)

class BinanceFuturesAdapter(ExchangeAdapter):
    """
    Órdenes reales en Binance USDⓈ-M Futures **TESTNET** vía CCXT.
    No maneja streaming; úsalo junto con un WS externo (binance_ws) para precios/last.
    """
    name = "binance_futures_usdm_testnet"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        leverage: int = 5,
    ):
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")

        self.leverage = leverage
        self.testnet = testnet

        # Cliente de Futuros USDⓈ-M
        self.rest = ccxt.binanceusdm({
            "apiKey": api_key or settings.binance_futures_api_key,
            "secret": api_secret or settings.binance_futures_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        # Importante: sandbox
        self.rest.set_sandbox_mode(testnet)

        # Asegurar modo one-way (no dual-side) y apalancamiento
        try:
            # Modo one-way (no posiciones long/short separadas)
            self.rest.set_position_mode(False)
        except Exception as e:
            log.debug("No se pudo fijar position_mode (one-way): %s", e)

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _ensure_leverage(self, symbol: str):
        try:
            market = self.rest.market(symbol)
            await self._to_thread(self.rest.set_leverage, self.leverage, market["symbol"])
        except Exception as e:
            log.debug("set_leverage falló (puede ya estar aplicado): %s", e)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        raise NotImplementedError("usa BinanceWSAdapter para stream de trades")

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, price: float | None = None, reduce_only: bool = False) -> dict:
        """
        side: 'buy'/'sell'
        type_: 'market'/'limit'
        qty: cantidad en unidades del activo base (p.ej. BTC) o contratos (en binanceusdm 1 contrato = 1 unidad base).
        """
        await self._ensure_leverage(symbol)

        order_type = "market" if type_.lower() == "market" else "limit"
        params = {"reduceOnly": reduce_only}

        try:
            resp = await self._to_thread(self.rest.create_order, symbol, order_type, side, qty, price, params)
            return {"status": "sent", "provider": "ccxt", "resp": resp}
        except Exception as e:
            log.error("Error create_order: %s", e)
            return {"status": "error", "error": str(e)}

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        try:
            if not symbol:
                return {"status": "error", "error": "symbol requerido por Binance para cancelar"}
            resp = await self._to_thread(self.rest.cancel_order, order_id, symbol)
            return {"status": "canceled", "resp": resp}
        except Exception as e:
            log.error("Error cancel_order: %s", e)
            return {"status": "error", "error": str(e)}
