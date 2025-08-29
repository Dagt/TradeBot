from __future__ import annotations

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import book_vacuum, liquidity_gap


PARAM_INFO = {
    "vacuum_threshold": "Umbral para detectar vacíos de liquidez",
    "gap_threshold": "Umbral para detectar gaps de liquidez",
    "tp_pct": "Take profit porcentual",
    "sl_pct": "Stop loss porcentual",
    "max_hold": "Barras máximas en posición",
    "vol_window": "Ventana para calcular la volatilidad",
    "dynamic_thresholds": "Ajustar umbrales según volatilidad",
}


class LiquidityEvents(Strategy):
    """React to liquidity vacuum and gap events.

    A wipe on the ask side triggers a buy signal while a wipe on the bid side
    results in a sell signal.  If no vaciado is detected, the strategy checks
    for large gaps between the first and second level of the book and trades in
    the direction of the gap.

    The strategy maintains an internal state and exits positions based on take
    profit, stop loss or maximum holding period.  Thresholds for detecting
    vacuums and gaps can be dynamically adjusted according to recent price
    volatility to increase the frequency of events during turbulent markets.
    """

    name = "liquidity_events"

    def __init__(
        self,
        vacuum_threshold: float = 0.5,
        gap_threshold: float = 1.0,
        tp_pct: float = 0.01,
        sl_pct: float = 0.005,
        max_hold: int = 30,
        vol_window: int = 20,
        dynamic_thresholds: bool = True,
    ):
        self.vacuum_threshold = vacuum_threshold
        self.gap_threshold = gap_threshold
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_hold = max_hold
        self.vol_window = vol_window
        self.dynamic_thresholds = dynamic_thresholds

        self.position: str | None = None
        self.entry_price: float = 0.0
        self.bars_in_position: int = 0

    def _mid_prices(self, df: pd.DataFrame) -> pd.Series:
        bid = df["bid_px"].apply(lambda x: x[0])
        ask = df["ask_px"].apply(lambda x: x[0])
        return (bid + ask) / 2

    def _vol_adjust(self, series: pd.Series, base: float) -> float:
        if not self.dynamic_thresholds:
            return base
        returns = series.pct_change().fillna(0)
        vol = returns.rolling(self.vol_window).std().iloc[-1]
        if pd.isna(vol):
            vol = 0.0
        return base / (1 + vol)

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty", "bid_px", "ask_px"}
        if not needed.issubset(df.columns) or len(df) < 2:
            return None

        mid = self._mid_prices(df)
        last_price = mid.iloc[-1]

        # Handle open positions
        if self.position == "long":
            self.bars_in_position += 1
            if (
                last_price >= self.entry_price * (1 + self.tp_pct)
                or last_price <= self.entry_price * (1 - self.sl_pct)
                or self.bars_in_position >= self.max_hold
            ):
                self.position = None
                return Signal("sell", 1.0, reduce_only=True)
            return None
        if self.position == "short":
            self.bars_in_position += 1
            if (
                last_price <= self.entry_price * (1 - self.tp_pct)
                or last_price >= self.entry_price * (1 + self.sl_pct)
                or self.bars_in_position >= self.max_hold
            ):
                self.position = None
                return Signal("buy", 1.0, reduce_only=True)
            return None

        vac_thresh = self._vol_adjust(mid, self.vacuum_threshold)
        vac = book_vacuum(df[["bid_qty", "ask_qty"]], vac_thresh).iloc[-1]
        if vac > 0:
            self.position = "long"
            self.entry_price = last_price
            self.bars_in_position = 0
            return Signal("buy", 1.0)
        if vac < 0:
            self.position = "short"
            self.entry_price = last_price
            self.bars_in_position = 0
            return Signal("sell", 1.0)

        gap_thresh = self._vol_adjust(mid, self.gap_threshold)
        gap = liquidity_gap(df[["bid_px", "ask_px"]], gap_thresh).iloc[-1]
        if gap > 0:
            self.position = "long"
            self.entry_price = last_price
            self.bars_in_position = 0
            return Signal("buy", 1.0)
        if gap < 0:
            self.position = "short"
            self.entry_price = last_price
            self.bars_in_position = 0
            return Signal("sell", 1.0)
        return None
