"""Binance connector using CCXT REST and native websockets.

Examples
--------
>>> c = BinanceConnector()
>>> await c.place_order(
...     "BTC/USDT", "buy", "limit", 1.0,
...     price=30_000, post_only=True, time_in_force="GTC", iceberg_qty=0.1
... )

Limitations
-----------
Only spot markets are supported and advanced order types depend on
Binance's REST API availability. Post-only orders are translated via the
``timeInForce`` parameter using ``GTX``. Iceberg quantity requires a limit
order and may not be accepted on all symbols.
"""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook, Trade, Funding, OpenInterest
from ..execution.venue_adapter import translate_order_flags


class BinanceConnector(ExchangeConnector):
    name = "binance"

    WS_BASE = "wss://stream.binance.com:9443/ws"
    WS_BASE_TESTNET = "wss://testnet.binance.vision/ws"

    def _ws_base(self) -> str:
        """Return websocket base depending on CCXT testnet option."""
        client = getattr(self, "rest", None)
        options = getattr(client, "options", {}) if client else {}
        return self.WS_BASE_TESTNET if options.get("testnet") else self.WS_BASE

    def _ws_url(self, symbol: str) -> str:
        return f"{self._ws_base()}/{symbol.lower()}@depth"

    def _ws_subscribe(self, symbol: str) -> str:
        return json.dumps(
            {"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@depth"], "id": 1}
        )

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:
        data = json.loads(msg)
        bids = [(float(p), float(q)) for p, q in data.get("b", [])]
        asks = [(float(p), float(q)) for p, q in data.get("a", [])]
        return OrderBook(
            timestamp=datetime.utcnow(),
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )

    def _ws_trades_url(self, symbol: str) -> str:
        return f"{self._ws_base()}/{symbol.lower()}@trade"

    def _ws_trades_subscribe(self, symbol: str) -> str:
        return json.dumps(
            {"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@trade"], "id": 1}
        )

    def _parse_trade(self, msg: str, symbol: str) -> Trade:
        data = json.loads(msg)
        ts = data.get("T") or data.get("t") or 0
        if ts > 1e12:
            ts /= 1000
        return Trade(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            price=float(data.get("p", 0.0)),
            amount=float(data.get("q", 0.0)),
            side="sell" if data.get("m") else "buy",
        )

    async def fetch_funding(self, symbol: str) -> Funding:
        """Fetch the current funding rate for ``symbol``.

        This simply proxies to the :class:`ExchangeConnector` implementation
        which in turn delegates to the underlying CCXT client.  The method is
        defined explicitly so that callers can rely on its presence for the
        Binance connector.
        """

        return await super().fetch_funding(symbol)

    async def fetch_open_interest(self, symbol: str) -> OpenInterest:
        """Fetch open interest for ``symbol``.

        Binance does not expose a dedicated endpoint in this connector so we
        fall back to the generic CCXT helper provided by the base class.
        """

        return await super().fetch_open_interest(symbol)

    # ------------------------------------------------------------------
    # Trading helpers
    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        *,
        post_only: bool = False,
        time_in_force: str | None = None,
        iceberg_qty: float | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """Submit an order to Binance via CCXT.

        Parameters are normalised and translated using
        :func:`~tradingbot.execution.venue_adapter.translate_order_flags` so
        individual connectors do not need to duplicate the mapping logic.
        """

        params = translate_order_flags(
            self.name,
            post_only=post_only,
            time_in_force=time_in_force,
            iceberg_qty=iceberg_qty,
            reduce_only=reduce_only,
        )

        data = await self._rest_call(
            self.rest.create_order, symbol, type_, side, qty, price, params
        )
        return {
            "id": data.get("id") or data.get("orderId"),
            "status": data.get("status"),
            "symbol": data.get("symbol", symbol),
            "side": data.get("side", side),
            "price": float(data.get("price") or data.get("avgPrice") or 0.0),
            "amount": float(data.get("amount") or data.get("origQty") or 0.0),
            "raw": data,
        }

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        """Cancel an order and return a minimal response."""

        data = await self._rest_call(self.rest.cancel_order, order_id, symbol)
        return {
            "id": data.get("id") or data.get("orderId") or order_id,
            "status": data.get("status") or data.get("result"),
            "raw": data,
        }

    async def fetch_balance(self) -> dict:
        """Fetch account balances normalising totals to floats."""

        data = await self._rest_call(self.rest.fetch_balance)
        balances: dict[str, float] = {}
        for asset, info in (data or {}).items():
            if isinstance(info, dict) and "total" in info:
                try:
                    balances[asset] = float(info["total"])
                except (TypeError, ValueError):
                    continue
        return balances
