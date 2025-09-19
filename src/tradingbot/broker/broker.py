from __future__ import annotations

import asyncio
import inspect
from typing import Callable, Optional

from ..config import settings
from ..execution.order_types import Order
from ..utils.metrics import ORDER_SENT


class Broker:
    """Simple broker wrapping an exchange adapter.

    Provides convenience helpers for limit orders with basic time-in-force
    handling, partial fills and re-quoting through optional callbacks.
    """

    def __init__(self, adapter, maker_fee_bps: float | None = None):
        self.adapter = adapter
        self.maker_fee_bps = (
            float(maker_fee_bps)
            if maker_fee_bps is not None
            else getattr(adapter, "maker_fee_bps", settings.maker_fee_bps)
        )
        self.passive_rebate_bps = getattr(settings, "passive_rebate_bps", 0.0)

    def update_last_price(self, symbol: str, px: float) -> None:
        """Update last price on the underlying adapter if supported."""
        upd = getattr(self.adapter, "update_last_price", None)
        if upd:
            upd(symbol, px)

    def equity(self) -> float:
        """Return account equity if the adapter exposes it."""
        eq = getattr(self.adapter, "equity", None)
        if callable(eq):
            try:
                return float(eq())
            except Exception:
                return 0.0
        bal = getattr(self.adapter, "account", None)
        if bal is not None and hasattr(bal, "equity"):
            return float(getattr(bal, "equity", 0.0))
        return 0.0

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        tif: str = "GTC|PO",
        on_partial_fill: Optional[Callable[[Order, dict], str | None]] = None,
        on_order_expiry: Optional[Callable[[Order, dict], str | None]] = None,
        signal_ts: float | None = None,
    ) -> dict:
        """Place a limit order respecting ``tif`` semantics.

        Parameters
        ----------
        symbol: str
            Trading pair, e.g. ``"BTC/USDT"``.
        side: str
            ``"buy"`` or ``"sell"``.
        price: float
            Limit price.
        qty: float
            Order quantity.
        tif: str, optional
            Time-in-force flags. Supports ``GTC``/``IOC``/``FOK`` and optional
            ``PO`` for post-only.  A ``GTD:<seconds>`` flag will cancel and
            replace the remaining quantity after the given number of seconds.
        on_partial_fill: callable, optional
            Callback executed when an order is partially filled. Should return
            ``"re_quote"`` to place the remaining quantity again. Any other
            value skips re-quoting.
        on_order_expiry: callable, optional
            Callback executed when an order expires (``GTD``). Should return
            ``"re_quote"`` to place the remaining quantity again. Any other
            value cancels the remainder.

        Returns
        -------
        dict
            Dictionary with keys ``status``, ``filled_qty``, ``pending_qty``,
            ``order_id`` and ``price`` describing the final state of the order
            or fallback order.
        """

        time_in_force = "GTC"
        post_only = False
        expiry = None
        for part in str(tif).split("|"):
            p = part.strip().upper()
            if p in {"GTC", "IOC", "FOK"}:
                time_in_force = p
            elif p == "PO":
                post_only = True
            elif p.startswith("GTD"):
                time_in_force = "GTD"
                try:
                    expiry = float(p.split(":", 1)[1])
                except (IndexError, ValueError):
                    expiry = None

        filled = 0.0
        remaining = qty
        attempts = 0
        max_attempts = int(getattr(settings, "requote_attempts", 5))
        last_res: dict = {}

        order = Order(
            symbol=symbol,
            side=side,
            type_="limit",
            qty=qty,
            price=price,
            post_only=post_only,
            time_in_force=time_in_force,
        )

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            kwargs = dict(
                symbol=symbol,
                side=side,
                type_="limit",
                qty=remaining,
                price=price,
                post_only=post_only,
                time_in_force=time_in_force,
            )
            if signal_ts is not None:
                sig = inspect.signature(self.adapter.place_order)
                params = sig.parameters
                if (
                    "signal_ts" in params
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                ):
                    kwargs["signal_ts"] = signal_ts
            ORDER_SENT.inc()
            res = await self.adapter.place_order(**kwargs)
            qty_filled = float(
                res.get("qty") or res.get("filled") or res.get("filled_qty") or 0.0
            )
            filled += qty_filled
            pending_reported = res.get("pending_qty")
            if pending_reported is not None:
                try:
                    remaining = max(float(pending_reported), 0.0)
                    res["pending_qty"] = remaining
                except (TypeError, ValueError):
                    remaining = max(remaining - qty_filled, 0.0)
            else:
                remaining = max(remaining - qty_filled, 0.0)
                res.setdefault("pending_qty", remaining)
            order.pending_qty = remaining
            res.setdefault("filled_qty", qty_filled)
            last_res = res

            # Fully filled or adapter rejected/canceled the order
            status = res.get("status")
            if remaining <= 0 or status in {"canceled", "rejected"}:
                break

            # Handle partial fills
            if qty_filled > 0 and remaining > 0:
                action = on_partial_fill(order, res) if on_partial_fill else None
                if action in {"re_quote", "requote", "re-quote"}:
                    continue
                break

            # Handle expiry/re-quote
            if expiry is not None and status not in {"canceled", "rejected"}:
                await asyncio.sleep(expiry)
                try:
                    await self.adapter.cancel_order(res.get("order_id"), symbol)
                except Exception:
                    pass
                action = on_order_expiry(order, res) if on_order_expiry else "re_quote"
                if action in {"re_quote", "requote", "re-quote"}:
                    price = order.price or price
                    continue
                break

            # Nothing else to do (no expiry and not fully filled)
            break

        return {
            "status": last_res.get("status"),
            "filled_qty": filled,
            "pending_qty": remaining,
            "order_id": last_res.get("order_id"),
            "price": last_res.get("price", price),
        }

    async def place_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        signal_ts: float | None = None,
    ) -> dict:
        """Place a market order and track submission metrics."""

        kwargs = {
            "symbol": symbol,
            "side": side,
            "type_": "market",
            "qty": qty,
        }
        if signal_ts is not None:
            kwargs["signal_ts"] = signal_ts
        ORDER_SENT.inc()
        return await self.adapter.place_order(**kwargs)
