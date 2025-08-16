"""Execution router with venue selection and basic metrics."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .order_types import Order
from ..utils.metrics import (
    SLIPPAGE,
    ORDER_LATENCY,
    MAKER_TAKER_RATIO,
    FILL_COUNT,
    QUEUE_POSITION,
)
from ..storage import timescale

log = logging.getLogger(__name__)


class ExecutionRouter:
    """Route orders to the best venue and track execution metrics."""

    def __init__(self, adapters: Iterable, storage_engine=None):
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
        self._engine = storage_engine

    # ------------------------------------------------------------------
    async def best_venue(self, order: Order):
        """Select venue based on spread, fees and latency."""

        def venue_score(adapter) -> float | None:
            state = getattr(adapter, "state", None)
            book = getattr(state, "order_book", {}).get(order.symbol) if state else None
            last = getattr(state, "last_px", {}).get(order.symbol) if state else None
            if book and book.get("bids") and book.get("asks"):
                bid = book["bids"][0][0]
                ask = book["asks"][0][0]
                spread = ask - bid
                price = ask if order.side == "buy" else bid
            elif last is not None:
                price = last
                spread = float("inf")
            else:
                return None

            fee_bps = getattr(adapter, "fee_bps", 0.0)
            fee = price * fee_bps / 10000.0
            latency = getattr(adapter, "latency", 0.0)
            direction = 1 if order.side == "buy" else -1
            score = direction * price + fee + spread / 2 + latency * 1e-6
            return score

        selected = None
        best_score = None
        for adapter in self.adapters.values():
            score = venue_score(adapter)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                selected = adapter
        if selected is None:
            selected = next(iter(self.adapters.values()))
        return selected

    # ------------------------------------------------------------------
    async def execute(self, order: Order, algo: str | None = None, **algo_kwargs) -> dict | list[dict]:
        """Execute an order using optional execution algorithms.

        Parameters
        ----------
        order:
            Order to be routed.
        algo:
            Optional execution algorithm name (``twap``, ``vwap`` or ``pov``).
        **algo_kwargs:
            Additional parameters forwarded to the chosen algorithm.

        Returns
        -------
        dict | list[dict]
            Result(s) from the adapter or algorithm.
        """

        if algo:
            from . import algos

            a = algo.lower()
            delay = float(algo_kwargs.get("delay", 0.0))
            if a == "twap":
                slices = int(algo_kwargs.get("slices", 1))
                children = algos.twap(order, slices)
            elif a == "vwap":
                volumes = algo_kwargs.get("volumes", [])
                children = algos.vwap(order, volumes)
            elif a == "pov":
                participation_rate = float(algo_kwargs["participation_rate"])
                trades = algo_kwargs.get("trades", [])
                children = algos.pov(order, trades, participation_rate)
            else:
                raise ValueError(f"unknown algo: {algo}")

            results: List[dict] = []
            for i, child in enumerate(children):
                res = await self.execute(child)
                results.append(res)
                if delay and i < len(children) - 1:
                    await asyncio.sleep(delay)
            return results

        adapter = await self.best_venue(order)
        venue = getattr(adapter, "name", "unknown")
        log.info("Routing order %s via %s", order, venue)

        state = getattr(adapter, "state", None)
        book = getattr(state, "order_book", {}).get(order.symbol) if state else None
        est_slippage = None
        queue_pos = None
        if book and book.get("bids") and book.get("asks"):
            levels: List[Tuple[float, float]] = (
                book["asks"] if order.side == "buy" else book["bids"]
            )
            top_px, top_qty = levels[0]
            # queue position relative to existing size at best level
            queue_pos = top_qty / (top_qty + order.qty) if top_qty else 0.0
            remaining = order.qty
            cost = 0.0
            for px, qty in levels:
                take = min(remaining, qty)
                cost += px * take
                remaining -= take
                if remaining <= 0:
                    break
            if remaining == 0:
                avg_px = cost / order.qty
                est_slippage = (avg_px - top_px) / top_px * 10000
                QUEUE_POSITION.labels(symbol=order.symbol, side=order.side).observe(queue_pos)

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
        if est_slippage is not None:
            res.setdefault("est_slippage_bps", est_slippage)
        if queue_pos is not None:
            res.setdefault("queue_position", queue_pos)
        if res.get("status") == "filled":
            FILL_COUNT.labels(symbol=order.symbol, side=order.side).inc()
            if self._engine is not None:
                try:
                    timescale.insert_order(
                        self._engine,
                        strategy="router",
                        exchange=venue,
                        symbol=order.symbol,
                        side=order.side,
                        type_=order.type_,
                        qty=float(order.qty),
                        px=res.get("price"),
                        status=res.get("status", "unknown"),
                        ext_order_id=res.get("order_id"),
                        notes=None,
                    )
                except Exception:
                    pass

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
        # Propagate venue so downstream components can track positions per exchange
        res.setdefault("venue", venue)
        return res


__all__ = ["ExecutionRouter"]

