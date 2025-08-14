# src/tradingbot/adapters/binance_futures.py
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

class BinanceFuturesAdapter(ExchangeAdapter):
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

        self.rest = ccxt.binanceusdm({
            "apiKey": api_key or settings.binance_futures_api_key,
            "secret": api_secret or settings.binance_futures_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.rest.set_sandbox_mode(testnet)

        # Cache de metadatos/filters
        self.meta = ExchangeMeta.binanceusdm_testnet(
            api_key or settings.binance_futures_api_key,
            api_secret or settings.binance_futures_api_secret
        )
        try:
            self.meta.load_markets()
        except Exception as e:
            log.warning("load_markets falló: %s", e)

        try:
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
            log.debug("set_leverage falló: %s", e)

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        raise NotImplementedError("usa BinanceWSAdapter para stream de trades")

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, reduce_only: bool = False, mark_price: float | None = None) -> Dict[str, Any]:
        """
        Envía una orden y retorna un dict con campos estandarizados:
        {
          "status": "filled"/"sent"/"error",
          "resp": <payload crudo>,
          "fill_price": float|None,
          "fee_usdt": float|None,
          "ext_order_id": str|None
        }
        """
        try:
            params = {"reduceOnly": reduce_only} if reduce_only else {}
            # IMPORTANTE: en testnet/ccxt, a veces el fill llega como market order inmediatamente.
            order = await self.rest.create_order(
                symbol.replace("/", ""),  # o símbolo según mapping que uses
                type_.lower(),
                side.lower(),
                qty,
                None,  # price para market = None
                params
            )
            # Extraer id
            ext_id = order.get("id") or order.get("orderId") or None

            # Intentar leer fills/avgPrice
            fill_px = None
            fee_usdt = None

            # ccxt: a veces trae 'average' y 'fee'
            # Nota: en binance futures el 'fee' puede venir como {cost, currency}. Convertimos a USDT si currency es USDT.
            try:
                if "info" in order and isinstance(order["info"], dict):
                    info = order["info"]
                    # average fill (si existe)
                    fill_px = float(info.get("avgPrice") or info.get("averagePrice") or info.get("price") or 0) or None
                    # fees: puede venir en 'fills' o en 'info'
                    fills = info.get("fills") or []
                    if isinstance(fills, list) and fills:
                        # sumamos fees en USDT si están
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
                # fallback a campos "average" de ccxt normalizados
                if fill_px is None and order.get("average"):
                    fill_px = float(order["average"])
                fee = order.get("fee")
                if fee and isinstance(fee, dict) and (fee.get("currency") or "").upper() in ("USDT", "BUSD"):
                    fee_usdt = float(fee.get("cost"))

            except Exception as e:
                log.debug("No se pudieron extraer fill/fee de la respuesta: %s", e)

            # Si no tenemos fill_px, usa mark_price como aproximación
            if fill_px is None and mark_price is not None:
                fill_px = float(mark_price)

            status = "filled" if fill_px is not None else "sent"
            return {"status": status, "resp": order, "fill_price": fill_px, "fee_usdt": fee_usdt, "ext_order_id": ext_id}
        except Exception as e:
            log.error("Error placing futures order: %s", e)
            return {"status": "error", "error": str(e), "resp": {}, "fill_price": None, "fee_usdt": None, "ext_order_id": None}