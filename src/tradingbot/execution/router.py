"""Execution router with venue selection and basic metrics.

The router can handle multiple exchange adapters, pick the venue with the
lowest estimated slippage and optionally run simple execution algorithms
such as TWAP, VWAP and POV. It also records Prometheus metrics for order
latency and maker/taker ratios.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Iterable, List, Sequence, Tuple

from .order_types import Order
from ..adapters.base import ExchangeAdapter
from ..utils.metrics import ORDER_LATENCY, SLIPPAGE, MAKER_TAKER_RATIO

log = logging.getLogger(__name__)


class ExecutionRouter:
    """Route orders to the best available venue.

    Parameters
    ----------
    adapters:
        Single adapter or iterable of adapters. The router will choose the
        venue with the smallest estimated slippage.
    """

    def __init__(self, adapters: ExchangeAdapter | Sequence[ExchangeAdapter]):
        if isinstance(adapters, Sequence):
            self.adapters: List[ExchangeAdapter] = list(adapters)
        else:  # allow single adapter for backwards compatibility
            self.adapters = [adapters]
        self._maker_taker: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Venue selection and slippage estimation
    # ------------------------------------------------------------------
    def estimate_slippage(self, order: Order, market_price: float | None) -> float:
        """Return absolute slippage ratio for the given order.

        The result is expressed as a fraction (e.g. ``0.0005`` = 5 bps).
        When market price information is not available the function returns
        ``0.0``.
        """

        if market_price is None or order.price is None:
            return 0.0
        return abs(market_price - order.price) / market_price

    def _adapter_price(self, adapter: ExchangeAdapter, symbol: str) -> float | None:
        state = getattr(adapter, "state", None)
        if state and hasattr(state, "last_px"):
            return state.last_px.get(symbol)
        getter = getattr(adapter, "get_price", None)
        if callable(getter):
            try:
                return getter(symbol)
            except Exception:  # pragma: no cover - adapter dependent
                return None
        return None

    async def best_venue(self, order: Order) -> Tuple[ExchangeAdapter, float]:
        """Select the adapter with the lowest estimated slippage."""

        best = self.adapters[0]
        best_slip = float("inf")
        for adapter in self.adapters:
            px = self._adapter_price(adapter, order.symbol)
            slip = self.estimate_slippage(order, px)
            if slip < best_slip:
                best = adapter
                best_slip = slip
        return best, best_slip if best_slip != float("inf") else 0.0

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _record_maker_taker(self, venue: str, is_maker: bool) -> None:
        counts = self._maker_taker.setdefault(venue, {"maker": 0, "taker": 0})
        counts["maker" if is_maker else "taker"] += 1
        ratio = counts["maker"] / max(counts["taker"], 1)
        MAKER_TAKER_RATIO.labels(venue=venue).set(ratio)

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------
    async def execute(self, order: Order, algo: str | None = None, **algo_kwargs):
        """Execute an order optionally using an execution algorithm.

        Parameters
        ----------
        order:
            The order to route.
        algo:
            Optional name of an execution algorithm (``"twap"``, ``"vwap"`` or
            ``"pov"``). If omitted, the order is sent directly to the selected
            venue.
        algo_kwargs:
            Parameters forwarded to the algorithm constructor.
        """

        adapter, slip = await self.best_venue(order)
        if slip:
            SLIPPAGE.labels(symbol=order.symbol, side=order.side).observe(slip * 10000)

        start = perf_counter()
        if algo:
            # Ensure sub-orders go to the chosen venue
            child_router = ExecutionRouter(adapter)
            algo_lower = algo.lower()
            if algo_lower == "twap":
                from .algos import TWAP

                exec_algo = TWAP(child_router, **algo_kwargs)
                result = await exec_algo.execute(order)
            elif algo_lower == "vwap":
                from .algos import VWAP

                exec_algo = VWAP(child_router, **algo_kwargs)
                result = await exec_algo.execute(order)
            elif algo_lower == "pov":
                from .algos import POV

                trades = algo_kwargs.pop("trades")
                exec_algo = POV(child_router, **algo_kwargs)
                result = await exec_algo.execute(order, trades)
            else:
                raise ValueError(f"Unknown algo: {algo}")
        else:
            result = await adapter.place_order(
                symbol=order.symbol,
                side=order.side,
                type_=order.type_,
                qty=order.qty,
                price=order.price,
                post_only=order.post_only,
                time_in_force=order.time_in_force,
                iceberg_qty=order.iceberg_qty,
            )

        latency = perf_counter() - start
        ORDER_LATENCY.labels(venue=adapter.name).observe(latency)
        self._record_maker_taker(adapter.name, order.post_only)
        log.info("Executed order on %s in %.4fs", adapter.name, latency)
        return result

