"""OKX connector using CCXT REST and native websockets.

Examples
--------
>>> c = OKXConnector()
>>> await c.place_order(
...     "BTC-USDT", "buy", "limit", 1,
...     price=20_000, post_only=True, iceberg_qty=0.3
... )

Limitations
-----------
Advanced order types depend on instrument support. Iceberg quantity maps
to the ``iceberg`` parameter and may require specific account settings.
"""
from __future__ import annotations

import json
from datetime import datetime

from .base import ExchangeConnector, OrderBook, Trade, Funding, OpenInterest


class OKXConnector(ExchangeConnector):
    name = "okx"

    def _ws_url(self, symbol: str) -> str:
        return "wss://ws.okx.com:8443/ws/v5/public"

    def _ws_subscribe(self, symbol: str) -> str:
        return json.dumps({"op": "subscribe", "args": [{"channel": "books", "instId": symbol}]})

    def _parse_order_book(self, msg: str, symbol: str) -> OrderBook:
        data = json.loads(msg)
        arg_data = data.get("data", [{}])[0]
        bids = [(float(p), float(q)) for p, q in arg_data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in arg_data.get("asks", [])]
        return OrderBook(
            timestamp=datetime.utcnow(),
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
        )

    def _ws_trades_url(self, symbol: str) -> str:
        return "wss://ws.okx.com:8443/ws/v5/public"

    def _ws_trades_subscribe(self, symbol: str) -> str:
        return json.dumps({"op": "subscribe", "args": [{"channel": "trades", "instId": symbol}]})

    def _parse_trade(self, msg: str, symbol: str) -> Trade:
        data = json.loads(msg)
        trade = (data.get("data") or [{}])[0]
        ts = trade.get("ts") or trade.get("t") or 0
        if ts > 1e12:
            ts = int(ts) / 1000
        else:
            ts = int(ts)
        return Trade(
            timestamp=datetime.fromtimestamp(ts),
            exchange=self.name,
            symbol=symbol,
            price=float(trade.get("px", 0.0)),
            amount=float(trade.get("sz", 0.0)),
            side=trade.get("side", ""),
        )

    async def fetch_funding(self, symbol: str) -> Funding:
        """Retrieve funding data for ``symbol`` using CCXT."""

        return await super().fetch_funding(symbol)

    async def fetch_open_interest(self, symbol: str) -> OpenInterest:
        """Retrieve open interest for ``symbol`` using CCXT."""

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
    ) -> dict:
        """Place an order via the underlying CCXT client.

        Parameters are normalised to the exchange requirements before being
        forwarded.  ``time_in_force`` and ``post_only`` are mapped to the
        respective CCXT parameters when provided.
        """

        params: dict[str, object] = {}
        if time_in_force:
            params["timeInForce"] = time_in_force
        if post_only:
            # CCXT para OKX soporta ``postOnly`` booleano
            params["postOnly"] = True
        if iceberg_qty is not None:
            params["iceberg"] = iceberg_qty

        data = await self._rest_call(
            self.rest.create_order, symbol, type_, side, qty, price, params
        )
        return {
            "id": data.get("id") or data.get("orderId"),
            "status": data.get("status"),
            "symbol": data.get("symbol", symbol),
            "side": data.get("side", side),
            "price": float(data.get("price") or data.get("avgPx") or data.get("px") or 0.0),
            "amount": float(
                data.get("amount")
                or data.get("filled")
                or data.get("size")
                or data.get("sz")
                or 0.0
            ),
            "raw": data,
        }

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        """Cancel an order and return a minimal normalised response."""

        data = await self._rest_call(self.rest.cancel_order, order_id, symbol)
        return {
            "id": data.get("id") or data.get("orderId") or order_id,
            "status": data.get("status") or data.get("result"),
            "raw": data,
        }

    async def fetch_balance(self) -> dict:
        """Fetch account balances normalising numeric fields to floats."""

        data = await self._rest_call(self.rest.fetch_balance)
        balances: dict[str, float] = {}
        for asset, info in (data or {}).items():
            if isinstance(info, dict) and "total" in info:
                try:
                    balances[asset] = float(info["total"])
                except (TypeError, ValueError):
                    continue
        return balances
