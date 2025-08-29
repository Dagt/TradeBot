from __future__ import annotations

from typing import Sequence, Type

from .base import Strategy, Signal, record_signal_metrics


class CompositeSignals(Strategy):
    """Combine signals from multiple sub-strategies."""

    name = "composite_signals"

    def __init__(self, strategies: Sequence[tuple[Type[Strategy], dict]]):
        self.sub_strategies = [cls(**params) for cls, params in strategies]

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        buys = 0
        sells = 0
        for strat in self.sub_strategies:
            sig = strat.on_bar(bar)
            if sig is None:
                continue
            if sig.side == "buy":
                buys += 1
            elif sig.side == "sell":
                sells += 1
        if buys >= 2 and buys > sells:
            return Signal("buy", 1.0)
        if sells >= 2 and sells > buys:
            return Signal("sell", 1.0)
        if buys > len(self.sub_strategies) / 2:
            return Signal("buy", 1.0)
        if sells > len(self.sub_strategies) / 2:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
