from __future__ import annotations

from typing import Sequence, Type

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics


class CompositeSignals(Strategy):
    """Combine signals from multiple sub-strategies."""

    name = "composite_signals"

    def __init__(
        self,
        strategies: Sequence[tuple[Type[Strategy], dict]],
        *,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
    ):
        self.sub_strategies = [cls(**params) for cls, params in strategies]
        self.tp_bps = float(tp_bps)
        self.sl_bps = float(sl_bps)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

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
        buy = False
        sell = False
        if buys >= 2 and buys > sells:
            buy = True
        elif sells >= 2 and sells > buys:
            sell = True
        elif buys > len(self.sub_strategies) / 2:
            buy = True
        elif sells > len(self.sub_strategies) / 2:
            sell = True

        df = bar.get("window")
        price = None
        if isinstance(df, pd.DataFrame):
            col = "close" if "close" in df.columns else "price"
            if col in df.columns:
                price = float(df[col].iloc[-1])

        if self.pos_side == 0:
            if buy:
                self.pos_side = 1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("buy", 1.0)
            if sell:
                self.pos_side = -1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("sell", 1.0)
            return None

        self.hold_bars += 1
        exit_signal = (sell and self.pos_side > 0) or (buy and self.pos_side < 0)
        exit_tp = exit_sl = False
        if price is not None and self.entry_price is not None:
            pnl_bps = (
                (price - self.entry_price) / self.entry_price * 10000 * self.pos_side
            )
            exit_tp = pnl_bps >= self.tp_bps
            exit_sl = pnl_bps <= -self.sl_bps
        exit_time = self.hold_bars >= self.max_hold_bars
        if exit_signal or exit_tp or exit_sl or exit_time:
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None
