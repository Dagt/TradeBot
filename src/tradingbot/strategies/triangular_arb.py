from __future__ import annotations

from typing import Optional, Dict

from .base import Strategy, Signal
from .arbitrage_triangular import compute_edge


class TriangularArb(Strategy):
    """Simple triangular arbitrage strategy.

    Expects prices for three markets forming a triangular route. Parameters
    are passed via ``**kwargs``.

    Args:
        taker_fee_bps (float): Taker fee per leg in basis points. Default ``0``.
        buffer_bps (float): Safety buffer per leg in basis points. Default ``0``.
        threshold (float): Minimum net edge to emit a signal. Default ``0``.
    """

    name = "triangular_arb"

    def __init__(self, **kwargs):
        self.taker_fee_bps = float(kwargs.get("taker_fee_bps", 0.0))
        self.buffer_bps = float(kwargs.get("buffer_bps", 0.0))
        self.threshold = float(kwargs.get("threshold", 0.0))

    def on_bar(self, bar: Dict) -> Optional[Signal]:
        prices = bar.get("prices")
        if prices is None:
            prices = {k: bar.get(k) for k in ("bq", "mq", "mb")}
        edge = compute_edge(prices, self.taker_fee_bps, self.buffer_bps)
        if edge and edge.net > self.threshold:
            return Signal(edge.direction, edge.net)
        return Signal("flat", 0.0)
