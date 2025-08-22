# src/tradingbot/adapters/binance_ws.py
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Any

try:  # pragma: no cover - ccxt es opcional en tests
    import ccxt.async_support as ccxt  # type: ignore
except Exception:  # pragma: no cover - si falta ccxt
    ccxt = None

from .base import ExchangeAdapter
from ..core.symbols import normalize

log = logging.getLogger(__name__)

def _binance_symbol_stream(symbol: str) -> str:
    # Binance usa minúsculas y sin separador para streams (BTCUSDT -> btcusdt)
    return symbol.replace("/", "").lower() + "@trade"

class BinanceWSAdapter(ExchangeAdapter):
    """
    Solo streaming de trades vía WS (no órdenes). Útil para alimentar precios al paper broker.
    """
    name = "binance"

    def __init__(
        self,
        ws_base: str | None = None,
        rest: Any | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        super().__init__()
        # Mainnet: wss://stream.binance.com:9443/stream?streams=
        # (Si usas testnet de futures, configura desde settings; lo dejamos parametrizable)
        self.ws_base = ws_base or "wss://stream.binance.com:9443/stream?streams="

        # Adapter REST opcional. Puede pasarse una instancia lista o
        # construirla a partir de credenciales si ccxt está disponible.
        self.rest = rest
        if self.rest is None and (api_key or api_secret):
            if ccxt is None:
                raise RuntimeError("ccxt no está instalado")
            self.rest = ccxt.binance({
                "apiKey": api_key or "",
                "secret": api_secret or "",
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })

    async def stream_trades(self, symbol: str) -> AsyncIterator[dict]:
        stream = _binance_symbol_stream(normalize(symbol))
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or {}
            price = float(d.get("p")) if d.get("p") is not None else None
            qty = float(d.get("q")) if d.get("q") is not None else None
            ts_ms = d.get("T")
            side = "sell" if d.get("m") else "buy"
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            if price is not None:
                self.state.last_px[symbol] = price
            yield self.normalize_trade(symbol, ts, price, qty, side)

    async def stream_order_book(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        stream = _binance_symbol_stream(normalize(symbol)).replace("@trade", f"@depth{depth}@100ms")
        url = self.ws_base + stream
        async for raw in self._ws_messages(url):
            msg = json.loads(raw)
            d = msg.get("data") or msg
            bids = d.get("bids") or d.get("b") or []
            asks = d.get("asks") or d.get("a") or []
            bids_n = [[float(b[0]), float(b[1])] for b in bids]
            asks_n = [[float(a[0]), float(a[1])] for a in asks]
            ts_ms = d.get("E") or d.get("T")
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
            self.state.order_book[symbol] = {"bids": bids_n, "asks": asks_n}
            yield self.normalize_order_book(symbol, ts, bids_n, asks_n)

    stream_orderbook = stream_order_book

    async def stream_bba(self, symbol: str) -> AsyncIterator[dict]:
        """Emit best bid/ask quotes for ``symbol``."""

        async for ob in self.stream_order_book(symbol):
            bid = ob.get("bid_px", [None])[0]
            ask = ob.get("ask_px", [None])[0]
            yield {"symbol": symbol, "ts": ob.get("ts"), "bid": bid, "ask": ask}

    async def stream_book_delta(self, symbol: str, depth: int = 10) -> AsyncIterator[dict]:
        """Yield incremental order book updates for ``symbol``."""

        prev: dict | None = None
        async for ob in self.stream_order_book(symbol, depth=depth):
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

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para funding")
        while True:
            data = await self.fetch_funding(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def stream_open_interest(self, symbol: str) -> AsyncIterator[dict]:
        """Poll open interest updates via REST."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para open interest")
        while True:
            data = await self.fetch_oi(symbol)
            yield {"symbol": symbol, **data}
            await asyncio.sleep(60)

    async def fetch_funding(self, symbol: str):
        """Obtiene la tasa de funding actual para ``symbol``."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para funding")

        sym = normalize(symbol)
        method = getattr(self.rest, "fetchFundingRate", None)
        if method is None:  # pragma: no cover - depende de ccxt
            raise NotImplementedError("Funding no soportado")

        data = await self._request(method, sym)
        ts = data.get("timestamp") or data.get("time") or data.get("ts") or 0
        ts = int(ts)
        if ts > 1e12:
            ts //= 1000
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        rate = float(
            data.get("fundingRate") or data.get("rate") or data.get("lastFundingRate") or 0.0
        )
        return {"ts": ts_dt, "rate": rate}

    async def fetch_basis(self, symbol: str):
        """Calcula la *basis* actual para ``symbol``.

        Binance expone el precio de referencia (*indexPrice*) y el precio
        marcado (*markPrice*) mediante el endpoint ``premiumIndex`` de su API
        de futuros.  La diferencia entre ambos representa el basis
        implícito del contrato perpetuo.  Si no se dispone de un adaptador
        REST, se lanza ``NotImplementedError`` para dejar claro que el venue
        no está soportado.
        """

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para basis")

        sym = normalize(symbol)
        method = getattr(self.rest, "fapiPublicGetPremiumIndex", None)
        if method is None:  # pragma: no cover - depende de ccxt
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
        """Obtiene el *open interest* actual para ``symbol``."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para open interest")

        sym = normalize(symbol)
        method = getattr(self.rest, "fapiPublicGetOpenInterest", None)
        if method is None:  # pragma: no cover - depende de ccxt
            raise NotImplementedError("Open interest no soportado")

        data = await self._request(method, {"symbol": sym})
        ts_ms = int(data.get("time") or data.get("timestamp") or data.get("ts") or 0)
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        oi = float(data.get("openInterest") or data.get("open_interest") or 0.0)
        return {"ts": ts, "oi": oi}

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """Envía una orden usando el adaptador REST si está disponible."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para enviar órdenes")

        params: dict = {}
        if post_only:
            params["timeInForce"] = "GTX"
        elif time_in_force:
            params["timeInForce"] = time_in_force

        return await self._request(
            self.rest.create_order, symbol, type_, side, qty, price, params
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        """Cancela una orden existente vía REST."""

        if not self.rest:
            raise NotImplementedError("Se requiere adaptador REST para cancelar órdenes")

        return await self._request(self.rest.cancel_order, order_id, symbol)
