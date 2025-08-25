# src/tradingbot/adapters/binance_spot.py
from __future__ import annotations
import asyncio
import logging, time, uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional, Any, Dict

try:
    import ccxt.async_support as ccxt
except Exception:
    ccxt = None

from .base import ExchangeAdapter
from ..config import settings
from ..core.symbols import normalize
from ..market.exchange_meta import ExchangeMeta
from .binance_errors import parse_binance_error_code  # si no lo tienes, ver notas abajo
from ..execution.retry import with_retries, AmbiguousOrderError
from ..utils.secrets import validate_scopes

log = logging.getLogger(__name__)

class BinanceSpotAdapter(ExchangeAdapter):
    """
    Adapter de Binance Spot con soporte para modo real o testnet.
    Usa WS aparte para precios (ver SpotWSAdapter abajo o pasa mark_price).
    """
    name = "binance_spot"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        maker_fee_bps: float | None = None,
        taker_fee_bps: float | None = None,
    ):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")

        key = api_key or (settings.binance_testnet_api_key if testnet else settings.binance_api_key)
        secret = api_secret or (settings.binance_testnet_api_secret if testnet else settings.binance_api_secret)

        self.rest = ccxt.binance(
            {
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
            }
        )
        self.rest.options["defaultType"] = "spot"
        self.maker_fee_bps = float(
            maker_fee_bps
            if maker_fee_bps is not None
            else (
                settings.binance_spot_testnet_maker_fee_bps
                if testnet
                else settings.binance_spot_maker_fee_bps
            )
        )
        self.taker_fee_bps = float(
            taker_fee_bps
            if taker_fee_bps is not None
            else (
                settings.binance_spot_testnet_taker_fee_bps
                if testnet
                else settings.binance_spot_taker_fee_bps
            )
        )
        self.rest.set_sandbox_mode(testnet)

        # Advertir si faltan scopes necesarios o si hay permisos peligrosos
        validate_scopes(self.rest, log)

        meta_factory = ExchangeMeta.binance_spot_testnet if testnet else ExchangeMeta.binance_spot
        self.meta = meta_factory(key, secret)
        try:
            self.meta.load_markets()
        except Exception as e:
            log.warning("load_markets spot falló: %s", e)

        self.name = "binance_spot_testnet" if testnet else "binance_spot"

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        sym = normalize(symbol)
        while True:
            trades = await self._request(self.rest.fetch_trades, sym, limit=1)
            for t in trades or []:
                ts = datetime.fromtimestamp(t.get("timestamp", 0) / 1000, tz=timezone.utc)
                price = float(t.get("price"))
                qty = float(t.get("amount", 0))
                side = t.get("side") or ""
                self.state.last_px[symbol] = price
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
        reduce_only: bool = False,
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
        sym = normalize(symbol)
        while True:
            ob = await self._request(self.rest.fetch_order_book, sym)
            ts_ms = ob.get("timestamp")
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            bids = [[float(b[0]), float(b[1])] for b in ob.get("bids", [])]
            asks = [[float(a[0]), float(a[1])] for a in ob.get("asks", [])]
            self.state.order_book[symbol] = {"bids": bids, "asks": asks}
            yield self.normalize_order_book(symbol, ts, bids, asks)
            await asyncio.sleep(1)

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Emit best bid/ask quotes for ``symbol``."""

        async for ob in self.stream_order_book(symbol):
            bid = ob.get("bid_px", [None])[0]
            bid_qty = ob.get("bid_qty", [None])[0]
            ask = ob.get("ask_px", [None])[0]
            ask_qty = ob.get("ask_qty", [None])[0]
            yield {
                "symbol": symbol,
                "ts": ob.get("ts"),
                "bid_px": bid,
                "bid_qty": bid_qty,
                "ask_px": ask,
                "ask_qty": ask_qty,
            }

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Yield incremental order book updates for ``symbol``."""

        prev: dict | None = None
        async for ob in self.stream_order_book(symbol):
            curr_bids = list(zip(ob.get("bid_px", []), ob.get("bid_qty", [])))
            curr_asks = list(zip(ob.get("ask_px", []), ob.get("ask_qty", [])))
            if prev is None:
                delta_bids = curr_bids
                delta_asks = curr_asks
            else:
                pb = dict(zip(prev.get("bid_px", []), prev.get("bid_qty", [])))
                pa = dict(zip(prev.get("ask_px", []), prev.get("ask_qty", [])))
                delta_bids = [[p, q] for p, q in curr_bids if pb.get(p) != q]
                delta_bids += [[p, 0.0] for p in pb.keys() - {p for p, _ in curr_bids}]
                delta_asks = [[p, q] for p, q in curr_asks if pa.get(p) != q]
                delta_asks += [[p, 0.0] for p in pa.keys() - {p for p, _ in curr_asks}]
            prev = ob
            yield {
                "symbol": symbol,
                "ts": ob.get("ts"),
                "bid_px": [p for p, _ in delta_bids],
                "bid_qty": [q for _, q in delta_bids],
                "ask_px": [p for p, _ in delta_asks],
                "ask_qty": [q for _, q in delta_asks],
            }

    async def stream_funding(self, symbol: str) -> AsyncIterator[dict]:
        """Poll funding rate updates via REST."""

        while True:
            data = await self.fetch_funding(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Poll open interest for ``symbol`` using the futures REST API."""

        while True:
            data = await self.fetch_oi(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def fetch_funding(self, symbol: str):
        """Return current funding rate for ``symbol``.

        Binance expone la tasa de funding del contrato perpetuo equivalente en
        ``fapi/v1/fundingRate``.  Consultamos ese endpoint y normalizamos la
        respuesta a ``{"ts": datetime, "rate": float}``.
        """

        sym = normalize(symbol)
        method = getattr(self.rest, "fapiPublicGetFundingRate", None)
        if method is None:
            raise NotImplementedError("Funding no soportado")

        data = await self._request(method, {"symbol": sym, "limit": 1})
        item = data[0] if isinstance(data, list) and data else data
        ts_ms = int(
            item.get("fundingTime")
            or item.get("timestamp")
            or item.get("time")
            or 0
        )
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rate = float(item.get("fundingRate") or item.get("rate") or 0.0)
        return {"ts": ts, "rate": rate}

    async def fetch_basis(self, symbol: str):
        """Return the basis (mark - index) for ``symbol``.

        El endpoint ``fapi/v1/premiumIndex`` expone tanto el ``indexPrice``
        como el ``markPrice`` del contrato perpetuo.  La diferencia entre ambos
        es la *basis*.
        """

        sym = normalize(symbol)
        method = getattr(self.rest, "fapiPublicGetPremiumIndex", None)
        if method is None:
            raise NotImplementedError("Basis no soportado")

        data = await self._request(method, {"symbol": sym})
        ts_ms = int(data.get("time") or data.get("timestamp") or 0)
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        mark_px = float(data.get("markPrice") or data.get("mark_price") or 0.0)
        index_px = float(data.get("indexPrice") or data.get("index_price") or 0.0)
        basis = mark_px - index_px
        return {"ts": ts, "basis": basis}

    async def fetch_oi(self, symbol: str):
        """Fetch futures open interest for the given spot ``symbol``.

        Binance only exposes open interest on the futures API.  We therefore
        call the ``fapiPublicGetOpenInterest`` endpoint and normalise the
        result into ``{"ts": datetime, "oi": float}`` so downstream code can
        store it directly.
        """

        sym = normalize(symbol)
        method = getattr(self.rest, "fapiPublicGetOpenInterest", None)
        if method is None:
            raise NotImplementedError("Open interest no soportado")

        data = await self._request(method, {"symbol": sym})
        ts_ms = int(data.get("time", 0))
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(data.get("openInterest", 0.0))
        return {"ts": ts, "oi": oi}

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        symbol_ex = symbol and normalize(symbol)
        return await self._request(self.rest.cancel_order, order_id, symbol_ex)
