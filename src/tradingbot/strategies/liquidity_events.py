from __future__ import annotations

import re

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import book_vacuum, liquidity_gap
from ..filters.liquidity import LiquidityFilterManager


PARAM_INFO = {
    "vacuum_threshold": "Umbral para detectar vacíos de liquidez",
    "gap_threshold": "Umbral para detectar gaps de liquidez",
    "vol_window": "Ventana para calcular la volatilidad",
    "dynamic_thresholds": "Ajustar umbrales según volatilidad",
}


liquidity = LiquidityFilterManager()


class LiquidityEvents(Strategy):
    """React to liquidity vacuum and gap events.

    A wipe on the ask side triggers a buy signal while a wipe on the bid side
    results in a sell signal. If no vaciado is detected, the strategy checks
    for large gaps between the first and second level of the book and trades in
    the direction of the gap.

    The strategy can optionally leverage a ``risk_service`` to size positions
    and manage exits. When no risk service is supplied it simply emits entry
    signals. The detection thresholds may be adjusted dynamically according to
    recent price volatility to increase the frequency of events during turbulent
    markets.
    """

    name = "liquidity_events"

    def __init__(
        self,
        vacuum_threshold: float = 0.5,
        gap_threshold: float = 1.0,
        vol_window: int = 20,
        dynamic_thresholds: bool = True,
        **kwargs,
    ):
        self.vacuum_threshold = vacuum_threshold
        self.gap_threshold = gap_threshold
        self.vol_window = vol_window
        self.dynamic_thresholds = dynamic_thresholds
        self.risk_service = kwargs.get("risk_service")

    def _mid_prices(self, df: pd.DataFrame) -> pd.Series:
        bid = df["bid_px"].apply(lambda x: x[0])
        ask = df["ask_px"].apply(lambda x: x[0])
        return (bid + ask) / 2

    def _effective_window(self, bar: dict) -> int:
        """Return volatility window adjusted for short timeframes.

        Timeframes between ``1m`` and ``5m`` use half of ``vol_window`` to
        increase sensitivity in fast markets.
        """

        tf = str(bar.get("timeframe", ""))
        m = re.match(r"^(\d+)m$", tf)
        if m:
            minutes = int(m.group(1))
            if 1 <= minutes <= 5:
                return max(5, self.vol_window // 2)
        return self.vol_window

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty", "bid_px", "ask_px"}
        if not needed.issubset(df.columns) or len(df) < 2:
            return None

        mid = self._mid_prices(df)
        last_price = float(mid.iloc[-1]) if len(mid) else 0.0

        window = self._effective_window(bar)
        sig: Signal | None = None
        if self.dynamic_thresholds:
            vol = mid.rolling(window).std().iloc[-1]
            if pd.isna(vol) or last_price == 0:
                vol = 0.0
            vac_thresh = (vol / last_price) * self.vacuum_threshold if last_price else self.vacuum_threshold
            gap_thresh = vol * self.gap_threshold
        else:
            vac_thresh = self.vacuum_threshold
            gap_thresh = self.gap_threshold

        vac = book_vacuum(df[["bid_qty", "ask_qty"]], vac_thresh).iloc[-1]
        if vac > 0:
            sig = Signal("buy", 1.0)
        elif vac < 0:
            sig = Signal("sell", 1.0)
        else:
            gap = liquidity_gap(df[["bid_px", "ask_px"]], gap_thresh).iloc[-1]
            if gap > 0:
                sig = Signal("buy", 1.0)
            elif gap < 0:
                sig = Signal("sell", 1.0)

        if sig is not None and self.risk_service is not None:
            qty = self.risk_service.calc_position_size(sig.strength, last_price)
            atr_val = bar.get("atr") or bar.get("volatility")
            stop = self.risk_service.initial_stop(last_price, sig.side, atr_val)
            self.trade = {
                "side": sig.side,
                "entry_price": last_price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
            }

        return self.finalize_signal(bar, last_price, sig)
