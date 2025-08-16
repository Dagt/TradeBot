"""Execution router with venue selection and basic metrics."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, Iterable

from .order_types import Order
from ..utils.metrics import SLIPPAGE, ORDER_LATENCY, MAKER_TAKER_RATIO

log = logging.getLogger(__name__)


class ExecutionRouter:
    """Route orders to the best venue and track execution metrics."""

    def __init__(self, adapters: Iterable):
        if isinstance(adapters, dict):
            self.adapters: Dict[str, object] = adapters  # type: ignore[assignment]
        elif isinstance(adapters, (list, tuple, set)):
            self.adapters = {
                getattr(ad, "name", f"venue{i}"): ad for i, ad in enumerate(adapters)
            }
        else:  # single adapter
            self.adapters = {getattr(adapters, "name", "default"): adapters}

        self._maker_counts = defaultdict(int)
        self._taker_counts = defaultdict(int)

    # ------------------------------------------------------------------
    async def best_venue(self, order: Order):
        """Select venue with best last price for the order side."""
        selected = None
        best_px = None
        for adapter in self.adapters.values():
            price = getattr(getattr(adapter, "state", None), "last_px", {}).get(order.symbol)
            if price is None:
                continue
            if order.side == "buy":
                if best_px is None or price < best_px:
                    best_px = price
                    selected = adapter
            else:
                if best_px is None or price > best_px:
                    best_px = price
                    selected = adapter
        if selected is None:
            selected = next(iter(self.adapters.values()))
        return selected

    # ------------------------------------------------------------------
    async def execute(self, order: Order) -> dict:
        adapter = await self.best_venue(order)
        venue = getattr(adapter, "name", "unknown")
        log.info("Routing order %s via %s", order, venue)

        start = time.monotonic()
        kwargs = dict(
            symbol=order.symbol,
            side=order.side,
            type_=order.type_,
            qty=order.qty,
            price=order.price,
            post_only=order.post_only,
            time_in_force=order.time_in_force,
        )
        if order.iceberg_qty is not None:
            kwargs["iceberg_qty"] = order.iceberg_qty

        res = await adapter.place_order(**kwargs)
        latency = time.monotonic() - start
        ORDER_LATENCY.labels(venue=venue).observe(latency)

        exec_price = res.get("price")
        expected = order.price
        if expected is None:
            expected = getattr(getattr(adapter, "state", None), "last_px", {}).get(order.symbol)
        if exec_price is not None and expected:
            bps = (exec_price - expected) / expected * 10000
            if bps != 0:
                SLIPPAGE.labels(symbol=order.symbol, side=order.side).observe(bps)

        maker = order.post_only or (
            order.type_.lower() == "limit" and order.time_in_force not in {"IOC", "FOK"}
        )
        if maker:
            self._maker_counts[venue] += 1
        else:
            self._taker_counts[venue] += 1
        maker_c = self._maker_counts[venue]
        taker_c = self._taker_counts[venue]
        ratio = maker_c / taker_c if taker_c else float(maker_c)
        MAKER_TAKER_RATIO.labels(venue=venue).set(ratio)

        return res


__all__ = ["ExecutionRouter"]

