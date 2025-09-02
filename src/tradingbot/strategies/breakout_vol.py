import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..utils.rolling_quantile import RollingQuantileCache

PARAM_INFO = {
    "lookback": "Ventana para medias y desviación estándar",
    "volatility_factor": "Factor para dimensionar según volatilidad",
}


class BreakoutVol(Strategy):
    """Ruptura por volatilidad con umbrales autocalibrados.

    El multiplicador del canal y la volatilidad mínima se estiman como
    percentiles recientes, de modo que la estrategia se adapta a las
    condiciones cambiantes del mercado sin necesidad de ajustar parámetros
    manualmente.
    """

    name = "breakout_vol"

    # Percentiles dependientes del timeframe para mínimos de volatilidad y
    # dimensionar el multiplicador.  Valores más bajos en timeframes cortos
    # permiten generar más señales sin relajar el control de riesgo.
    _VOL_QUANTILES = {"1m": 0.1, "5m": 0.2}
    _MULT_QUANTILES = {"1m": 0.7, "5m": 0.8}
    _DEFAULT_VOL_Q = 0.2
    _DEFAULT_MULT_Q = 0.8

    def __init__(self, **kwargs):
        self.lookback = kwargs.get("lookback", 10)
        self.volatility_factor = kwargs.get("volatility_factor", 0.02)
        self.risk_service = kwargs.get("risk_service")
        tf = kwargs.get("timeframe", "3m")
        self._vol_quantile = self._VOL_QUANTILES.get(tf, self._DEFAULT_VOL_Q)
        self._mult_quantile = self._MULT_QUANTILES.get(tf, self._DEFAULT_MULT_Q)
        self.timeframe = tf
        # Valores dinámicos calculados en ``on_bar``.
        self.mult = 1.0
        self.min_volatility = 0.0
        self._rq = RollingQuantileCache()

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.lookback + 1:
            return None
        closes = df["close"]
        mean = closes.rolling(self.lookback).mean().iloc[-1]
        std_series = closes.rolling(self.lookback).std().dropna()
        std = float(std_series.iloc[-1])
        window = min(len(std_series), self.lookback * 5)
        symbol = bar.get("symbol", "")
        if window >= self.lookback and std > 0.0:
            rq_std = self._rq.get(
                symbol,
                "std",
                window=self.lookback * 5,
                q=self._mult_quantile,
                min_periods=self.lookback,
            )
            mult_quant = float(rq_std.update(std))
            self.mult = mult_quant / std
        else:
            self.mult = 1.0

        last = float(closes.iloc[-1])
        upper = mean + self.mult * std
        lower = mean - self.mult * std

        returns = closes.pct_change().dropna()
        vol_series = returns.rolling(self.lookback).std().dropna()
        vol = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
        vol_bps = vol * 10000
        window = min(len(vol_series), self.lookback * 5)
        if window >= self.lookback * 2:
            rq_vol = self._rq.get(
                symbol,
                "vol", 
                window=self.lookback * 5,
                q=self._vol_quantile,
                min_periods=self.lookback * 2,
            )
            vol_quant = float(rq_vol.update(vol))
            rq_vol_bps = self._rq.get(
                symbol,
                "vol_bps",
                window=self.lookback * 5,
                q=self._vol_quantile,
                min_periods=self.lookback * 2,
            )
            vol_bps_quant = float(rq_vol_bps.update(vol_bps))
            self.min_volatility = vol_bps_quant
            if vol < vol_quant or vol_bps < vol_bps_quant:
                return None
            # Ajuste dinámico del multiplicador cuando la volatilidad es baja
            low_vol_threshold = vol_quant * 1.5
            if vol < low_vol_threshold and self.mult > 0.0:
                factor = max(0.5, vol / low_vol_threshold)
                self.mult *= factor
        else:
            self.min_volatility = 0.0

        size = max(0.0, min(1.0, vol_bps * self.volatility_factor))

        side: str | None = None
        if last > upper:
            side = "buy"
        elif last < lower:
            side = "sell"
        if side is None:
            return self.finalize_signal(bar, last, None)
        sig = Signal(side, size)

        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(size, last)
            stop = self.risk_service.initial_stop(last, side, vol)
            self.trade = {
                "side": side,
                "entry_price": last,
                "qty": qty,
                "stop": stop,
                "atr": vol,
            }

        return self.finalize_signal(bar, last, sig)
