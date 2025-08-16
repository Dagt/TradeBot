# src/tradingbot/adapters/binance_futures.py
from __future__ import annotations
import asyncio
import os
import logging, time, uuid
from typing import AsyncIterator, Optional, Any, Dict

try:
    import ccxt
except Exception:
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings
from ..market.exchange_meta import ExchangeMeta
from ..execution.normalize import adjust_order
from .binance_errors import parse_binance_error_code
from ..execution.retry import with_retries, AmbiguousOrderError

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
        self.taker_fee_bps = float(os.getenv("TRADING_TAKER_FEE_BPS_FUT", "5.0"))
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

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        mark_price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
        iceberg_qty: float | None = None,
    ):
        """
        UM Futures: idempotencia + retries + reconciliación.
        """
        symbol_ex = symbol.replace("/", "")
        side_u = side.upper()
        type_u = type_.upper()
        qty = float(qty)
        cid = client_order_id or f"TBT-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

        async def _send():
            try:
                send_type = type_u
                params = {
                    "symbol": symbol_ex,
                    "side": side_u,
                    "type": send_type,
                    "quantity": qty,
                    "newClientOrderId": cid,
                }
                if iceberg_qty:
                    params["icebergQty"] = float(iceberg_qty)
                if reduce_only:
                    params["reduceOnly"] = True
                if send_type == "LIMIT":
                    if price is None:
                        raise ValueError("LIMIT requiere price")
                    params["price"] = float(price)
                    tif = time_in_force or ("GTX" if post_only else "GTC")
                    params["timeInForce"] = tif
                if post_only and send_type != "LIMIT":
                    params["timeInForce"] = "GTX"

                resp = await self.rest.futures_order_new(**params)
                return {
                    "status": "sent",
                    "ext_order_id": resp.get("orderId"),
                    "client_order_id": resp.get("clientOrderId") or cid,
                    "fill_price": None,
                    "raw": resp,
                }
            except Exception as e:
                code = parse_binance_error_code(e)
                transient = { -1001, -1003, -1021, -1105, None }
                fatal = { -1013, -2010, -2011, -2019 }
                ambiguous = { -1001, None }

                if code in fatal:
                    log.error("[FUT] Orden fatal %s %s: %s", symbol, params, e)
                    raise
                if code in ambiguous:
                    raise AmbiguousOrderError(f"Ambiguous fut order send: {e}", candidate_client_order_id=cid)

                log.warning("[FUT] Transient error %s → retry: %s", code, e)
                raise

        def _on_retry(attempt, exc):
            log.warning("[FUT] retry #%d %s %s qty=%.8f reduceOnly=%s: %s",
                        attempt, side_u, symbol, qty, reduce_only, exc)

        try:
            out = await with_retries(_send, max_attempts=5, on_retry=_on_retry)
            return out
        except AmbiguousOrderError as ae:
            # Reconciliación por clientOrderId
            try:
                q = await self.rest.futures_order_get(symbol=symbol_ex, origClientOrderId=ae.client_order_id)
                return {
                    "status": q.get("status", "NEW"),
                    "ext_order_id": q.get("orderId"),
                    "client_order_id": q.get("clientOrderId"),
                    "fill_price": None,
                    "raw": q,
                }
            except Exception as e2:
                try:
                    lst = await self.rest.futures_all_orders(symbol=symbol_ex, limit=5)
                    for o in lst or []:
                        if o.get("clientOrderId") == ae.client_order_id:
                            return {
                                "status": o.get("status", "NEW"),
                                    "ext_order_id": o.get("orderId"),
                                    "client_order_id": o.get("clientOrderId"),
                                    "fill_price": None,
                                    "raw": o,
                                }
                except Exception:
                    pass
                log.error("[FUT] Reconciliación falló (%s): %s", ae.client_order_id, e2)
                return {
                    "status": "unknown",
                    "ext_order_id": None,
                    "client_order_id": ae.client_order_id,
                    "fill_price": None,
                    "raw": {"error": "reconcile_failed"},
                }