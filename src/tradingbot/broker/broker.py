from __future__ import annotations

import asyncio
from datetime import datetime

from ..config import settings


class Broker:
    """Simple broker wrapping an exchange adapter.

    Provides convenience helpers for limit orders with basic time-in-force
    handling and accounting of partial fills.
    """

    def __init__(self, adapter, maker_fee_bps: float | None = None):
        self.adapter = adapter
        self.maker_fee_bps = (
            float(maker_fee_bps)
            if maker_fee_bps is not None
            else getattr(adapter, "maker_fee_bps", settings.maker_fee_bps)
        )
        self.passive_rebate_bps = getattr(settings, "passive_rebate_bps", 0.0)

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        tif: str = "GTC|PO",
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
        total_time = 0.0
        remaining = qty
        attempts = 0
        max_attempts = 5 if expiry is not None else 1
        last_res: dict = {}

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            start = datetime.utcnow()
            res = await self.adapter.place_order(
                symbol,
                side,
                "limit",
                remaining,
                price,
                post_only=post_only,
                time_in_force=time_in_force,
            )
            qty_filled = float(res.get("qty") or res.get("filled") or 0.0)
            filled += qty_filled
            remaining = max(remaining - qty_filled, 0.0)
            end = datetime.utcnow()
            total_time += (end - start).total_seconds()
            last_res = res
            if remaining <= 0 or expiry is None or res.get("status") in {"canceled", "rejected"}:
                break
            await asyncio.sleep(expiry)
            total_time += expiry
            try:
                await self.adapter.cancel_order(res.get("order_id"), symbol)
            except Exception:
                pass

        last_res.setdefault("filled_qty", filled)
        last_res.setdefault("time_in_book", total_time)
        return last_res
