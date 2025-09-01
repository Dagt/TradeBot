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
        window = bar.get("window")
        if window is None or "close" not in window:
            return None
        price = float(window["close"].iloc[-1])
        buys = 0
        sells = 0
        df = bar.get("window")
        if df is None:
            return None
        col = (
            "close"
            if "close" in df.columns
            else "price" if "price" in df.columns else None
        )
        if col is None:
            return None
        price = float(df[col].iloc[-1])
        for strat in self.sub_strategies:
            sig = strat.on_bar(bar)
            if sig is None:
                continue
            if sig.side == "buy":
                buys += 1
            elif sig.side == "sell":
                sells += 1

        result = None
        if buys >= 2 and buys > sells:
            result = Signal("buy", 1.0)
        elif sells >= 2 and sells > buys:
            result = Signal("sell", 1.0)
        elif buys > len(self.sub_strategies) / 2:
            result = Signal("buy", 1.0)
        elif sells > len(self.sub_strategies) / 2:
            result = Signal("sell", 1.0)

        if result is not None:
            result.limit_price = price
        return result
