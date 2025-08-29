from __future__ import annotations

import pandas as pd

from .base import StatefulStrategy, Signal, record_signal_metrics
from ..data.features import book_vacuum, liquidity_gap


class LiquidityEvents(StatefulStrategy):
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
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
        max_hold_bars: int = 0,
    ) -> None:
        super().__init__(tp_pct=tp_pct, sl_pct=sl_pct, max_hold_bars=max_hold_bars)
        self.vacuum_threshold = vacuum_threshold
        self.gap_threshold = gap_threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty", "bid_px", "ask_px"}
        if not needed.issubset(df.columns) or len(df) < 2:
            return None

        price = float(df.get("close", df["bid_px"].apply(lambda x: x[0]).iloc[-1]))
        exit_sig = self.check_exit(price)
        if exit_sig:
            return exit_sig

        vac = book_vacuum(df[list({"bid_qty", "ask_qty"})], self.vacuum_threshold).iloc[-1]
        if self.pos_side is None:
            if vac > 0:
                return self.open_position("buy", price)
            if vac < 0:
                return self.open_position("sell", price)

            gap = liquidity_gap(df[list({"bid_px", "ask_px"})], self.gap_threshold).iloc[-1]
            if gap > 0:
                return self.open_position("buy", price)
            if gap < 0:
                return self.open_position("sell", price)
        return None
