# src/tradingbot/adapters/binance_spot.py
from __future__ import annotations
import asyncio
import os
import logging, time, uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional, Any, Dict

try:
    import ccxt
except Exception:
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings
from ..market.exchange_meta import ExchangeMeta
from .binance_errors import parse_binance_error_code  # si no lo tienes, ver notas abajo
from ..execution.retry import with_retries, AmbiguousOrderError
from ..utils.secrets import validate_scopes

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
        self.taker_fee_bps = float(os.getenv("TRADING_TAKER_FEE_BPS", "10.0"))
        self.rest.set_sandbox_mode(testnet)

        # Advertir si faltan scopes necesarios o si hay permisos peligrosos
        validate_scopes(self.rest, log)

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
        sym = self.normalize_symbol(symbol)
        while True:
            trades = await self._request(self.rest.fetch_trades, sym, limit=1)
            for t in trades or []:
                ts = datetime.fromtimestamp(t.get("timestamp", 0) / 1000, tz=timezone.utc)
                price = float(t.get("price"))
                qty = float(t.get("amount", 0))
                side = t.get("side") or ""
                yield self.normalize_trade(symbol, ts, price, qty, side)
            await asyncio.sleep(1)

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        mark_price: float | None = None,
        client_order_id: str | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
    ):
        """
        Envía orden con idempotencia (NEW CLIENT ORDER ID) + retries.
        Reconciliación posterior si el error es ambiguo (timeout / desconexión).
        """
        symbol_ex = symbol.replace("/", "")
        side_u = side.upper()
        type_u = type_.upper()
        qty = float(qty)

        # Idempotencia: aseguramos uno por llamada si no viene
        cid = client_order_id or f"TBT-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

        async def _send():
            try:
                send_type = "LIMIT_MAKER" if post_only and type_u == "LIMIT" else type_u
                params = {
                    "symbol": symbol_ex,
                    "side": side_u,
                    "type": send_type,
                    "quantity": qty,
                    "newClientOrderId": cid,
                }
                if send_type in {"LIMIT", "LIMIT_MAKER"}:
                    if price is None:
                        raise ValueError("LIMIT requiere price")
                    params["price"] = float(price)
                    params["timeInForce"] = time_in_force or "GTC"
                # Llamada REST real
                resp = await self.rest.order_new(**params)
                return {
                    "status": "sent",
                    "ext_order_id": resp.get("orderId"),
                    "client_order_id": resp.get("clientOrderId") or cid,
                    "fill_price": None,
                    "raw": resp,
                }
            except Exception as e:
                code = parse_binance_error_code(e)  # puede retornar None
                # Errores transitorios típicos → reintento
                transient = { -1001, -1003, -1021, -1105, None }
                # Errores de validación → no reintentar
                fatal = { -1013, -2010, -2011, -2019 }
                # Cuando no sabemos si entró (timeout, conexión), marcamos ambiguo
                ambiguous = { -1001, None }

                if code in fatal:
                    log.error("[SPOT] Orden fatal %s %s: %s", symbol, params, e)
                    raise

                if code in ambiguous:
                    # Puede haber sido aceptada: reconciliar afuera
                    raise AmbiguousOrderError(f"Ambiguous spot order send: {e}", candidate_client_order_id=cid)

                # Transitorio: reintentar
                log.warning("[SPOT] Transient error %s → retry: %s", code, e)
                raise

        def _on_retry(attempt, exc):
            log.warning("[SPOT] retry #%d %s %s qty=%.8f: %s", attempt, side_u, symbol, qty, exc)

        try:
            out = await with_retries(_send, max_attempts=5, on_retry=_on_retry)
            return out
        except AmbiguousOrderError as ae:
            # Reconciliación: buscamos por clientOrderId
            try:
                q = await self.rest.order_get(symbol=symbol_ex, origClientOrderId=ae.client_order_id)
                return {
                    "status": q.get("status", "NEW"),
                    "ext_order_id": q.get("orderId"),
                    "client_order_id": q.get("clientOrderId"),
                    "fill_price": None,
                    "raw": q,
                }
            except Exception as e2:
                # Último recurso: consulta por últimas órdenes
                try:
                    lst = await self.rest.all_orders(symbol=symbol_ex, limit=5)
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
                log.error("[SPOT] Reconciliación falló (%s): %s", ae.client_order_id, e2)
                # devolvemos un “estado desconocido” para que arriba maneje si hace falta
                return {
                    "status": "unknown",
                    "ext_order_id": None,
                    "client_order_id": ae.client_order_id,
                    "fill_price": None,
                    "raw": {"error": "reconcile_failed"},
                }

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        sym = self.normalize_symbol(symbol)
        while True:
            ob = await self._request(self.rest.fetch_order_book, sym)
            ts_ms = ob.get("timestamp")
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            bids = [[float(b[0]), float(b[1])] for b in ob.get("bids", [])]
            asks = [[float(a[0]), float(a[1])] for a in ob.get("asks", [])]
            yield self.normalize_order_book(symbol, ts, bids, asks)
            await asyncio.sleep(1)

    async def fetch_funding(self, symbol: str):
        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:
            raise NotImplementedError("Funding no soportado en spot")
        data = await self._request(method, sym)
        ts = data.get("timestamp") or data.get("time") or data.get("ts") or 0
        ts = int(ts)
        if ts > 1e12:
            ts /= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        rate = float(data.get("fundingRate") or data.get("rate") or data.get("value") or 0.0)
        return {"ts": ts_dt, "rate": rate}

    async def fetch_oi(self, symbol: str):
        """Fetch futures open interest for the given spot ``symbol``.

        Binance only exposes open interest on the futures API.  We therefore
        call the ``fapiPublicGetOpenInterest`` endpoint and normalise the
        result into ``{"ts": datetime, "oi": float}`` so downstream code can
        store it directly.
        """

        sym = self.normalize_symbol(symbol)
        method = getattr(self.rest, "fapiPublicGetOpenInterest", None)
        if method is None:
            raise NotImplementedError("Open interest no soportado")

        data = await self._request(method, {"symbol": sym})
        ts_ms = int(data.get("time", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(data.get("openInterest", 0.0))
        return {"ts": ts, "oi": oi}

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        symbol_ex = symbol and self.normalize_symbol(symbol)
        return await self._request(self.rest.cancel_order, order_id, symbol_ex)