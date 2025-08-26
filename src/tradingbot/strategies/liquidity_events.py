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

    def __init__(self, vacuum_threshold: float = 0.5, gap_threshold: float = 1.0):
        self.vacuum_threshold = vacuum_threshold
        self.gap_threshold = gap_threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty", "bid_px", "ask_px"}
        if not needed.issubset(df.columns) or len(df) < 2:
            return None

        vac = book_vacuum(df[list({"bid_qty", "ask_qty"})], self.vacuum_threshold).iloc[-1]
        if vac > 0:
            return Signal("buy", 1.0, target_pct=1.0)
        if vac < 0:
            return Signal("sell", 1.0, target_pct=1.0)

        gap = liquidity_gap(df[list({"bid_px", "ask_px"})], self.gap_threshold).iloc[-1]
        if gap > 0:
            return Signal("buy", 1.0, target_pct=1.0)
        if gap < 0:
            return Signal("sell", 1.0, target_pct=1.0)
        return Signal("flat", 0.0, target_pct=0.0)
