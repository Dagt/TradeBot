import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager

PARAM_INFO = {
    "lookback": "Ventana para medias y desviación estándar",
    "volatility_factor": "Factor para dimensionar según volatilidad",
}


liquidity = LiquidityFilterManager()


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
    _VOL_QUANTILES = {"1m": 0.3, "3m": 0.25, "5m": 0.2}
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
        self.cooldown_bars = kwargs.get("cooldown_bars", 3 if tf in {"1m", "3m"} else 0)
        self.vol_ma_n = kwargs.get("volume_ma_n", 20)
        self._cooldown = 0
        self._last_rpnl = 0.0
        self._last_month: int | None = None
        # Valores dinámicos calculados en ``on_bar``.
        self.mult = 1.0
        self.min_volatility = 0.0
        self._rq = RollingQuantileCache()

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.lookback + 1:
            return None

        if self.risk_service is not None:
            rpnl = getattr(self.risk_service.pos, "realized_pnl", 0.0)
            if rpnl < self._last_rpnl and abs(self.risk_service.pos.qty) < 1e-9:
                self._cooldown = self.cooldown_bars
            self._last_rpnl = rpnl
        if self._cooldown > 0:
            self._cooldown -= 1
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

        high = df.get("high")
        low = df.get("low")
        atr_val = 0.0
        if high is not None and low is not None:
            prev_close = closes.shift()
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr_series = tr.rolling(self.lookback).mean().dropna()
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0

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
            self.mult = max(0.5, self.mult)
        else:
            self.min_volatility = 0.0

        size = max(0.0, min(1.0, vol_bps * self.volatility_factor))

        vol_ma = df["volume"].rolling(self.vol_ma_n).mean().iloc[-1]
        if self.timeframe in {"1m", "3m"} and df["volume"].iloc[-1] <= vol_ma:
            return self.finalize_signal(bar, last, None)

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
                "atr_price": atr_val,
            }

        symbol = bar.get("symbol", "")
        if self.timeframe in {"15m", "30m"} and self.risk_service is not None and symbol:
            trade = self.risk_service.get_trade(symbol)
            if trade:
                entry = trade.get("entry_price")
                atr = trade.get("atr_price", atr_val)
                if entry is not None and atr:
                    if trade.get("side") == "buy" and last < entry - 3 * atr:
                        return self.finalize_signal(bar, last, Signal("sell", 1.0))
                    if trade.get("side") == "sell" and last > entry + 3 * atr:
                        return self.finalize_signal(bar, last, Signal("buy", 1.0))
            ts = pd.to_datetime(bar.get("timestamp") or df.index[-1])
            month = ts.month
            if self._last_month is None:
                self._last_month = month
            elif month != self._last_month:
                self._last_month = month
                if trade:
                    side_exit = "sell" if trade.get("side") == "buy" else "buy"
                    return self.finalize_signal(bar, last, Signal(side_exit, 1.0))

        return self.finalize_signal(bar, last, sig)
