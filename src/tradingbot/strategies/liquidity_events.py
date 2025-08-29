from __future__ import annotations

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import book_vacuum, liquidity_gap


class LiquidityEvents(Strategy):
    """React to liquidity vacuum and gap events.

    A wipe on the ask side triggers a buy signal while a wipe on the bid side
    results in a sell signal.  If no vaciado is detected, the strategy checks
    for large gaps between the first and second level of the book and trades in
    the direction of the gap.
    """

    name = "liquidity_events"

    def __init__(
        self,
        vacuum_threshold: float = 0.5,
        gap_threshold: float = 1.0,
        *,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
    ):
        self.vacuum_threshold = vacuum_threshold
        self.gap_threshold = gap_threshold
        self.tp_bps = float(tp_bps)
        self.sl_bps = float(sl_bps)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty", "bid_px", "ask_px"}
        if not needed.issubset(df.columns) or len(df) < 2:
            return None

        vac = book_vacuum(df[list({"bid_qty", "ask_qty"})], self.vacuum_threshold).iloc[-1]
        gap = liquidity_gap(df[list({"bid_px", "ask_px"})], self.gap_threshold).iloc[-1]
        buy = vac > 0 or gap > 0
        sell = vac < 0 or gap < 0
        price = None
        if {"close"}.issubset(df.columns):
            price = float(df["close"].iloc[-1])

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
