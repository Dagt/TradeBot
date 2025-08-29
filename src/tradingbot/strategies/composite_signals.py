from __future__ import annotations

from typing import Sequence, Type

from .base import Strategy, Signal, record_signal_metrics


PARAM_INFO = {
    "strategies": "Lista de subestrategias y sus parámetros",
    "tp_pct": "Take profit porcentual",
    "sl_pct": "Stop loss porcentual",
}


class CompositeSignals(Strategy):
    """Combine signals from multiple sub-strategies."""

    name = "composite_signals"

    def __init__(
        self,
        strategies: Sequence[tuple[Type[Strategy], dict]],
        *,
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
    ):
        self.sub_strategies = [cls(**params) for cls, params in strategies]
        self.tp_pct = float(tp_pct)
        self.sl_pct = float(sl_pct)

    def _levels(self, side: str, price: float | None) -> dict[str, float | None]:
        tp = sl = None
        if price is not None:
            if side == "buy":
                if self.tp_pct > 0:
                    tp = price * (1 + self.tp_pct)
                if self.sl_pct > 0:
                    sl = price * (1 - self.sl_pct)
            else:  # sell
                if self.tp_pct > 0:
                    tp = price * (1 - self.tp_pct)
                if self.sl_pct > 0:
                    sl = price * (1 + self.sl_pct)
        return {"take_profit": tp, "stop_loss": sl}

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
        price = bar.get("close") or bar.get("price")
        if buys >= 2 and buys > sells:
            levels = self._levels("buy", price)
            return Signal("buy", 1.0, **levels)
        if sells >= 2 and sells > buys:
            levels = self._levels("sell", price)
            return Signal("sell", 1.0, **levels)
        if buys > len(self.sub_strategies) / 2:
            levels = self._levels("buy", price)
            return Signal("buy", 1.0, **levels)
        if sells > len(self.sub_strategies) / 2:
            levels = self._levels("sell", price)
            return Signal("sell", 1.0, **levels)
        return None
