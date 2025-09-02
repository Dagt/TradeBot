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
        df = bar.get("window")
        if df is None:
            return None

        tf = bar.get("timeframe")
        intraday = isinstance(tf, str) and tf.lower().endswith(("s", "m", "h"))

        # Requerir más historial para marcos intradía, ya que son más ruidosos.
        min_window = 20 if intraday else 1
        if len(df) < min_window:
            return None

        col = (
            "close"
            if "close" in df.columns
            else "price" if "price" in df.columns else None
        )
        if col is None:
            return None
        price = float(df[col].iloc[-1])

        # En timeframes intradía las subestrategias secundarias pesan la mitad
        # para mitigar el ruido. El primer elemento mantiene peso completo.
        if intraday:
            weights = [1.0] + [0.5] * (len(self.sub_strategies) - 1)
        else:
            weights = [1.0] * len(self.sub_strategies)

        buys = 0.0
        sells = 0.0
        for strat, weight in zip(self.sub_strategies, weights):
            sig = strat.on_bar(bar)
            if sig is None:
                continue
            if sig.side == "buy":
                buys += weight
            elif sig.side == "sell":
                sells += weight

        total = sum(weights)
        result = None
        if buys >= total / 2 and buys > sells:
            result = Signal("buy", 1.0)
        elif sells >= total / 2 and sells > buys:
            result = Signal("sell", 1.0)
        return self.finalize_signal(bar, price, result)
