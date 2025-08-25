# src/tradingbot/adapters/binance_futures.py
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
from ..execution.normalize import adjust_order
from ..execution.venue_adapter import translate_order_flags
from .binance_errors import parse_binance_error_code
from ..execution.retry import with_retries, AmbiguousOrderError
from ..utils.secrets import validate_scopes

log = logging.getLogger(__name__)

class BinanceFuturesAdapter(ExchangeAdapter):
    name = "binance_futures"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        leverage: int = 5,
        maker_fee_bps: float | None = None,
        taker_fee_bps: float | None = None,
    ):
        super().__init__()
        if ccxt is None:
            raise RuntimeError("ccxt no está instalado")

        self.leverage = leverage
        self.testnet = testnet

        if testnet:
            self.rest = ccxt.binanceusdm(
                {
                    "apiKey": api_key or settings.binance_futures_api_key,
                    "secret": api_secret or settings.binance_futures_api_secret,
                    "enableRateLimit": True,
                }
            )
        else:
            self.rest = ccxt.binanceusdm()
        self.rest.options["defaultType"] = "future"
        self.maker_fee_bps = float(
            maker_fee_bps
            if maker_fee_bps is not None
            else (
                settings.binance_futures_testnet_maker_fee_bps
                if testnet
                else settings.binance_futures_maker_fee_bps
            )
        )
        self.taker_fee_bps = float(
            taker_fee_bps
            if taker_fee_bps is not None
            else (
                settings.binance_futures_testnet_taker_fee_bps
                if testnet
                else settings.binance_futures_taker_fee_bps
            )
        )
        self.rest.set_sandbox_mode(testnet)

        # Advertir si faltan scopes necesarios o sobran permisos
        validate_scopes(self.rest, log)

        # Cache de metadatos/filters
        self.meta = ExchangeMeta.binanceusdm_testnet(
            api_key or settings.binance_futures_api_key,
            api_secret or settings.binance_futures_api_secret,
        )
        try:
            self.meta.load_markets()
        except Exception as e:
            log.warning("load_markets falló: %s", e)

        self.name = "binance_futures_testnet" if testnet else "binance_futures"

        try:
            self._configure_lock = asyncio.Lock()
            self._position_mode_configured = False
        except Exception:
            # Lock creation should not fail, but keep constructor safe
            self._configure_lock = None
            self._position_mode_configured = False

    async def _configure_mode(self):
        """Ensure one-way position mode is configured."""
        if getattr(self, "_position_mode_configured", False):
            return
        lock = getattr(self, "_configure_lock", None)
        if lock is None:
            # Lock not available, still attempt configuration
            try:
                await self.rest.set_position_mode(False)
            except Exception as e:
                log.debug("No se pudo fijar position_mode (one-way): %s", e)
            else:
                self._position_mode_configured = True
            return
        async with lock:
            if self._position_mode_configured:
                return
            try:
                await self.rest.set_position_mode(False)
            except Exception as e:
                log.debug("No se pudo fijar position_mode (one-way): %s", e)
            else:
                self._position_mode_configured = True

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        await self._configure_mode()
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
        reduce_only: bool = False,
        client_order_id: str | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
        iceberg_qty: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
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
                params = translate_order_flags(
                    self.name,
                    post_only=post_only,
                    time_in_force=time_in_force,
                    iceberg_qty=iceberg_qty,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                )
                params.update(
                    {
                        "symbol": symbol_ex,
                        "side": side_u,
                        "type": send_type,
                        "quantity": qty,
                        "newClientOrderId": cid,
                    }
                )
                if reduce_only:
                    params["reduceOnly"] = True
                if send_type == "LIMIT":
                    if price is None:
                        raise ValueError("LIMIT requiere price")
                    params["price"] = float(price)
                    params.setdefault("timeInForce", "GTC")

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

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        symbol_ex = symbol and normalize(symbol)
        try:
            return await self._request(self.rest.futures_cancel_order, symbol_ex, orderId=order_id)
        except Exception as e:  # pragma: no cover - logging only
            log.error("[FUT] cancel_order error: %s", e)
            raise

    async def stream_order_book(self, symbol: str) -> AsyncIterator[dict]:
        await self._configure_mode()
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
            ask = ob.get("ask_px", [None])[0]
            yield {"symbol": symbol, "ts": ob.get("ts"), "bid": bid, "ask": ask}

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
        await self._configure_mode()
        while True:
            data = await self.fetch_funding(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Poll open interest updates via REST."""
        await self._configure_mode()
        while True:
            data = await self.fetch_oi(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def fetch_funding(self, symbol: str):
        sym = normalize(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:
            raise NotImplementedError("Funding no soportado")
        data = await self._request(method, sym)
        ts = data.get("timestamp") or data.get("fundingTime") or data.get("time") or data.get("ts") or 0
        ts = int(ts)
        if ts > 1e12:
            ts /= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        rate = float(data.get("fundingRate") or data.get("rate") or data.get("value") or 0.0)
        return {"ts": ts_dt, "rate": rate}

    async def fetch_basis(self, symbol: str):
        """Return the basis (mark - index) for ``symbol``.

        Binance expone el ``markPrice`` y el ``indexPrice`` del contrato
        perpetuo mediante el endpoint ``premiumIndex`` de su API pública. La
        diferencia entre ambos valores se conoce como *basis*.  Si el método
        correspondiente no está disponible en el cliente REST se lanza un
        ``NotImplementedError`` para dejar claro que el venue no lo soporta.
        """

        sym = normalize(symbol)
        method = getattr(self.rest, "public_get_premiumindex", None)
        if method is None:
            raise NotImplementedError("Basis no soportado")

        data = await self._request(method, {"symbol": sym})
        ts = data.get("time") or data.get("timestamp") or data.get("ts") or 0
        ts = int(ts)
        if ts > 1e12:
            ts //= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        mark_px = float(data.get("markPrice") or data.get("mark_price") or 0.0)
        index_px = float(data.get("indexPrice") or data.get("index_price") or 0.0)
        basis = mark_px - index_px
        return {"ts": ts_dt, "basis": basis}

    async def fetch_oi(self, symbol: str):
        """Fetch current open interest for ``symbol``.

        Binance exposes the open interest via the futures REST API.  Even for
        the futures testnet this endpoint lives under ``fapiPublic``.  The
        response contains the current open interest value and a millisecond
        timestamp.  We normalise it into ``{"ts": datetime, "oi": float}`` so
        callers can persist it without further transformation.
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
