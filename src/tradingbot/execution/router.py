"""Execution router with venue selection and basic metrics."""

from __future__ import annotations

import asyncio
import logging
import time
import inspect
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .order_types import Order
from .slippage import impact_by_depth, queue_position
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
    """Route orders to the best venue and track execution metrics.

    A ``prefer`` flag can force maker or taker routing regardless of the
    order's ``post_only`` attribute.  When left as ``None`` the router keeps
    the previous behaviour of inferring maker vs taker from the order type.
    ``prefer`` accepts ``"maker"`` or ``"taker"``.
    """

    def __init__(self, adapters: Iterable, storage_engine=None, prefer: str | None = None):
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
        self._prefer = prefer
        self._last_maker = False

    # ------------------------------------------------------------------
    def _is_maker(self, order: Order) -> bool:
        maker = order.post_only or (
            order.type_.lower() == "limit" and order.time_in_force not in {"IOC", "FOK"}
        )
        if self._prefer == "maker":
            return True
        if self._prefer == "taker":
            return False
        return maker

    # ------------------------------------------------------------------
    async def best_venue(self, order: Order):
        """Select venue based on spread, fees and latency."""
        maker_pref = self._is_maker(order)

        def venue_score(adapter) -> tuple[float, bool] | None:
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

            maker = maker_pref
            fee_attr = "maker_fee_bps" if maker else "taker_fee_bps"
            fee_bps = getattr(adapter, fee_attr, getattr(adapter, "fee_bps", 0.0))
            fee = price * fee_bps / 10000.0
            latency = getattr(adapter, "latency", 0.0)
            direction = 1 if order.side == "buy" else -1
            score = direction * price + fee + spread / 2 + latency * 1e-6
            return score, maker

        selected = None
        best_score = None
        selected_maker = maker_pref
        for adapter in self.adapters.values():
            res = venue_score(adapter)
            if res is None:
                continue
            score, maker = res
            if best_score is None or score < best_score:
                best_score = score
                selected = adapter
                selected_maker = maker
        if selected is None:
            selected = next(iter(self.adapters.values()))
        self._last_maker = selected_maker
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

        maker = getattr(self, "_last_maker", self._is_maker(order))

        state = getattr(adapter, "state", None)
        book = getattr(state, "order_book", {}).get(order.symbol) if state else None
        est_slippage = None
        queue_pos = None
        if book and book.get("bids") and book.get("asks"):
            levels: List[Tuple[float, float]] = (
                book["bids"] if maker and order.side == "buy" else
                book["asks"] if maker and order.side == "sell" else
                book["asks"] if order.side == "buy" else book["bids"]
            )
            if maker:
                queue_pos = queue_position(order.qty, levels)
                QUEUE_POSITION.labels(symbol=order.symbol, side=order.side).observe(queue_pos)
            else:
                est_slippage = impact_by_depth(order.side, order.qty, levels)

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
            sig = inspect.signature(adapter.place_order)
            params = sig.parameters
            if (
                "iceberg_qty" in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            ):
                kwargs["iceberg_qty"] = order.iceberg_qty

        res = await adapter.place_order(**kwargs)
        latency = time.monotonic() - start
        ORDER_LATENCY.labels(venue=venue).observe(latency)
        if est_slippage is not None:
            res.setdefault("est_slippage_bps", est_slippage)
        if queue_pos is not None:
            res.setdefault("queue_position", queue_pos)
        fee_attr = "maker_fee_bps" if maker else "taker_fee_bps"
        fee_bps = getattr(adapter, fee_attr, getattr(adapter, "fee_bps", 0.0))
        res.setdefault("fee_type", "maker" if maker else "taker")
        res.setdefault("fee_bps", fee_bps)
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
                        notes={"fee_type": "maker" if maker else "taker", "fee_bps": fee_bps},
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

