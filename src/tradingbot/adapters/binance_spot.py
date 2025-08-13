# src/tradingbot/adapters/binance_spot.py
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
from ..market.exchange_meta import ExchangeMeta
from ..execution.normalize import adjust_order

log = logging.getLogger(__name__)

class BinanceSpotAdapter(ExchangeAdapter):
    """
    Órdenes reales en Binance SPOT **TESTNET** vía CCXT, con validación (tick/step/minNotional).
    Usa WS aparte para precios (ver SpotWSAdapter abajo o pasa mark_price).
    """
    name = "binance_spot_testnet"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")
        self.rest = ccxt.binance({
            "apiKey": api_key or settings.binance_api_key,
            "secret": api_secret or settings.binance_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.rest.set_sandbox_mode(testnet)

        self.meta = ExchangeMeta.binance_spot_testnet(
            api_key or settings.binance_api_key,
            api_secret or settings.binance_api_secret
        )
        try:
            self.meta.load_markets()
        except Exception as e:
            log.warning("load_markets spot falló: %s", e)

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        raise NotImplementedError("Usa SpotWSAdapter o pasa mark_price manual")

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, price: float | None = None, reduce_only: bool = False, mark_price: float | None = None) -> dict:
        await self._ensure_leverage(symbol)
        try:
            rules = self.meta.rules_for(symbol)
        except Exception as e:
            log.debug("No se pudieron obtener reglas; enviando sin normalizar: %s", e)
            rules = None

        if rules:
            ar = adjust_order(price if type_.lower()=="limit" else None,
                              qty,
                              mark_price or (price or 0.0),
                              rules,
                              side)
            if not ar.ok:
                return {"status": "rejected", "reason": ar.reason, "adj": {"price": ar.price, "qty": ar.qty, "notional": ar.notional}}
            qty = ar.qty
            if type_.lower() == "limit":
                price = ar.price

        order_type = "market" if type_.lower() == "market" else "limit"
        params = {"reduceOnly": reduce_only}

        async def _call():
            return await self._to_thread(self.rest.create_order, symbol, order_type, side, qty, price, params)

        try:
            resp = await with_retry(_call, retries=5, base_delay=0.4, max_delay=3.0)
            return {"status": "sent", "provider": "ccxt", "resp": resp}
        except Exception as e:
            log.error("Error create_order: %s", e)
            return {"status": "error", "error": str(e), "symbol": symbol, "side": side, "qty": qty, "price": price}
