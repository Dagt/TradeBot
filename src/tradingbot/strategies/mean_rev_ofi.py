import numpy as np
import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import calc_ofi, returns


PARAM_INFO = {
    "ofi_window": "Ventana para estadísticos de OFI",
    "zscore_threshold": "Z-score absoluto requerido",
    "vol_window": "Ventana para volatilidad de retornos",
    "vol_threshold": "Volatilidad máxima permitida",
    "min_volatility": "Volatilidad mínima reciente en bps",
    "config_path": "Ruta opcional de configuración",
}


class MeanRevOFI(Strategy):
    """Mean reversion strategy based on OFI z-score and volatility.

    Generates inverse signals: when buying pressure (positive OFI z-score)
    exceeds a threshold under low volatility, a ``sell`` signal is emitted and
    vice versa.

    Parameters
    ----------
    ofi_window : int, optional
        Lookback window for the OFI rolling statistics, by default ``5``.
    zscore_threshold : float, optional
        Absolute z-score required to trigger a trade, by default ``1.0``.
    vol_window : int, optional
        Window for the volatility estimation of log returns, by default ``20``.
    vol_threshold : float, optional
        Maximum volatility allowed to take a position, by default ``0.01``.
    """

    name = "mean_rev_ofi"

    def __init__(
        self,
        ofi_window: int = 5,
        zscore_threshold: float = 1.0,
        vol_window: int = 20,
        vol_threshold: float = 0.01,
        min_volatility: float = 0.0,
        *,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        params = {**load_params(config_path), **kwargs}
        self.risk_service = params.pop("risk_service", None)
        self.ofi_window = int(params.get("ofi_window", ofi_window))
        self.zscore_threshold = float(params.get("zscore_threshold", zscore_threshold))
        self.vol_window = int(params.get("vol_window", vol_window))
        self.vol_threshold = float(params.get("vol_threshold", vol_threshold))
        self.min_volatility = float(params.get("min_volatility", min_volatility))
        self.trade: dict | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else None

        if self.trade and self.risk_service and last_close is not None:
            self.risk_service.update_trailing(self.trade, last_close)
            trade_state = {**self.trade, "current_price": last_close}
            decision = self.risk_service.manage_position(trade_state)
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            if decision in {"scale_in", "scale_out"}:
                self.trade["strength"] = trade_state.get("strength", 1.0)
                return Signal(self.trade["side"], self.trade["strength"])
            return None

        needed = {"bid_qty", "ask_qty", "close"}
        min_len = max(self.ofi_window, self.vol_window) + 1
        if not needed.issubset(df.columns) or len(df) < min_len:
            return None

        ofi_series = calc_ofi(df[["bid_qty", "ask_qty"]])
        rolling_mean = ofi_series.rolling(self.ofi_window).mean()
        rolling_std = ofi_series.rolling(self.ofi_window).std(ddof=0).replace(0, np.nan)
        zscore = ((ofi_series - rolling_mean) / rolling_std).iloc[-1]

        vol = returns(df).rolling(self.vol_window).std().iloc[-1]
        vol_bps = vol * 10000

        if (
            pd.isna(zscore)
            or pd.isna(vol)
            or vol >= self.vol_threshold
            or vol_bps < self.min_volatility
        ):
            return None

        if zscore > self.zscore_threshold:
            side = "sell"
        elif zscore < -self.zscore_threshold:
            side = "buy"
        else:
            return None
        strength = 1.0
        if self.risk_service and last_close is not None:
            qty = self.risk_service.calc_position_size(strength, last_close)
            trade = {"side": side, "entry_price": last_close, "qty": qty, "strength": strength}
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(
                last_close, side, atr
            )
            trade["atr"] = atr
            self.risk_service.update_trailing(trade, last_close)
            self.trade = trade
        return Signal(side, strength)
