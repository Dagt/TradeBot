from __future__ import annotations

from typing import Sequence, Type

from .base import Strategy, Signal, record_signal_metrics


PARAM_INFO = {
    "strategies": "Lista de subestrategias y sus parámetros",
}


class CompositeSignals(Strategy):
    """Combine signals from multiple sub-strategies."""

    name = "composite_signals"

    def __init__(
        self,
        strategies: Sequence[tuple[Type[Strategy], dict]],
        *,
        risk_service=None,
    ):
        self.risk_service = risk_service
        self.sub_strategies = []
        for cls, params in strategies:
            try:
                self.sub_strategies.append(cls(risk_service=risk_service, **params))
            except TypeError:
                # Sub-strategy may not accept ``risk_service``; fall back to
                # instantiating without it to preserve backward compatibility.
                self.sub_strategies.append(cls(**params))

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        buys = 0
        sells = 0
        price = float(bar.get("close") or bar.get("price") or 0.0)
        for strat in self.sub_strategies:
            sig = strat.on_bar(bar)
            if sig is None:
                continue
            if sig.side == "buy":
                buys += 1
            elif sig.side == "sell":
                sells += 1
        if buys >= 2 and buys > sells:
            sig = Signal("buy", 1.0)
            sig.limit_price = price
            return sig
        if sells >= 2 and sells > buys:
            sig = Signal("sell", 1.0)
            sig.limit_price = price
            return sig
        if buys > len(self.sub_strategies) / 2:
            sig = Signal("buy", 1.0)
            sig.limit_price = price
            return sig
        if sells > len(self.sub_strategies) / 2:
            sig = Signal("sell", 1.0)
            sig.limit_price = price
            return sig
        return None
