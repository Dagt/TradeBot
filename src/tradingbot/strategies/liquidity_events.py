from __future__ import annotations

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import book_vacuum, liquidity_gap


PARAM_INFO = {
    "vacuum_threshold": "Umbral para detectar vacíos de liquidez",
    "gap_threshold": "Umbral para detectar gaps de liquidez",
    "vol_window": "Ventana para calcular la volatilidad",
    "dynamic_thresholds": "Ajustar umbrales según volatilidad",
}


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
        risk_service=None,
    ):
        self.vacuum_threshold = vacuum_threshold
        self.gap_threshold = gap_threshold
        self.vol_window = vol_window
        self.dynamic_thresholds = dynamic_thresholds
        self.risk_service = risk_service
        self.trade: dict | None = None

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

        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, last_price)
            decision = self.risk_service.manage_position({**self.trade, "current_price": last_price})
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            return None

        vac_thresh = self._vol_adjust(mid, self.vacuum_threshold)
        vac = book_vacuum(df[["bid_qty", "ask_qty"]], vac_thresh).iloc[-1]
        if vac > 0:
            side = "buy"
            strength = 1.0
            if self.risk_service:
                qty = self.risk_service.calc_position_size(strength, last_price)
                trade = {"side": side, "entry_price": float(last_price), "qty": qty, "strength": strength}
                atr = bar.get("atr") or bar.get("volatility")
                trade["stop"] = self.risk_service.initial_stop(last_price, side, atr)
                if atr is not None:
                    trade["atr"] = atr
                self.risk_service.update_trailing(trade, last_price)
                self.trade = trade
            return Signal(side, strength)
        if vac < 0:
            side = "sell"
            strength = 1.0
            if self.risk_service:
                qty = self.risk_service.calc_position_size(strength, last_price)
                trade = {"side": side, "entry_price": float(last_price), "qty": qty, "strength": strength}
                atr = bar.get("atr") or bar.get("volatility")
                trade["stop"] = self.risk_service.initial_stop(last_price, side, atr)
                if atr is not None:
                    trade["atr"] = atr
                self.risk_service.update_trailing(trade, last_price)
                self.trade = trade
            return Signal(side, strength)

        gap_thresh = self._vol_adjust(mid, self.gap_threshold)
        gap = liquidity_gap(df[["bid_px", "ask_px"]], gap_thresh).iloc[-1]
        if gap > 0:
            side = "buy"
            strength = 1.0
            if self.risk_service:
                qty = self.risk_service.calc_position_size(strength, last_price)
                trade = {"side": side, "entry_price": float(last_price), "qty": qty, "strength": strength}
                atr = bar.get("atr") or bar.get("volatility")
                trade["stop"] = self.risk_service.initial_stop(last_price, side, atr)
                if atr is not None:
                    trade["atr"] = atr
                self.risk_service.update_trailing(trade, last_price)
                self.trade = trade
            return Signal(side, strength)
        if gap < 0:
            side = "sell"
            strength = 1.0
            if self.risk_service:
                qty = self.risk_service.calc_position_size(strength, last_price)
                trade = {"side": side, "entry_price": float(last_price), "qty": qty, "strength": strength}
                atr = bar.get("atr") or bar.get("volatility")
                trade["stop"] = self.risk_service.initial_stop(last_price, side, atr)
                if atr is not None:
                    trade["atr"] = atr
                self.risk_service.update_trailing(trade, last_price)
                self.trade = trade
            return Signal(side, strength)
        return None
