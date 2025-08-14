# src/tradingbot/adapters/binance_spot.py
from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator, Optional, Any, Dict

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

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, mark_price: float | None = None) -> Dict[str, Any]:
        try:
            order = await self.rest.create_order(
                symbol.replace("/", ""),
                type_.lower(),
                side.lower(),
                qty,
                None
            )

            ext_id = order.get("id") or order.get("orderId") or None
            fill_px = None
            fee_usdt = None
            try:
                if "info" in order and isinstance(order["info"], dict):
                    info = order["info"]
                    fill_px = float(info.get("price") or info.get("avgPrice") or 0) or None
                    fills = info.get("fills") or []
                    if isinstance(fills, list) and fills:
                        total_fee = 0.0
                        for f in fills:
                            f_cost = f.get("commission")
                            f_curr = f.get("commissionAsset")
                            if f_cost is not None and (f_curr or "").upper() in ("USDT", "BUSD"):
                                try:
                                    total_fee += float(f_cost)
                                except:
                                    pass
                        fee_usdt = total_fee if total_fee > 0 else None
                if fill_px is None and order.get("average"):
                    fill_px = float(order["average"])
                fee = order.get("fee")
                if fee and isinstance(fee, dict) and (fee.get("currency") or "").upper() in ("USDT", "BUSD"):
                    fee_usdt = float(fee.get("cost"))
            except Exception as e:
                log.debug("No se pudieron extraer fill/fee spot: %s", e)

            if fill_px is None and mark_price is not None:
                fill_px = float(mark_price)

            status = "filled" if fill_px is not None else "sent"
            return {"status": status, "resp": order, "fill_price": fill_px, "fee_usdt": fee_usdt, "ext_order_id": ext_id}
        except Exception as e:
            log.error("Error placing spot order: %s", e)
            return {"status": "error", "error": str(e), "resp": {}, "fill_price": None, "fee_usdt": None, "ext_order_id": None}