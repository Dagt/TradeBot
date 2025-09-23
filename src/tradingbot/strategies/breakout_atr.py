import math
import pandas as pd
from .base import (
    Strategy,
    Signal,
    load_params,
    record_signal_metrics,
    timeframe_to_minutes,
)
from ..data.features import atr, keltner_channels
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager

PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "vol_quantile": "Percentil base para filtrar baja volatilidad (1m)",
    "offset_frac": "Fracción base del ATR usada para cruzar el mercado (1m)",
    "volume_factor": "Multiplicador de volumen mínimo requerido",
    "cooldown_bars": "Barras a esperar tras una pérdida",
}


liquidity = LiquidityFilterManager()


class BreakoutATR(Strategy):
    """Keltner breakout con filtros de volatilidad autocalibrados.

    El umbral de volatilidad y el ancho del canal se recalculan en cada
    barra usando percentiles recientes del ATR, por lo que el usuario no
    necesita ajustar parámetros adicionales para filtrar períodos de baja
    volatilidad. Las órdenes límite aplican un pequeño offset basado en el
    ATR, cruzan el último precio negociado para garantizar ejecuciones
    inmediatas o en la siguiente barra y lo incrementan de forma
    progresiva si la orden expira sin ejecutarse, buscando mejorar la tasa
    de ejecución sin requerir parámetros adicionales.
    """

    name = "breakout_atr"

    # Percentil usado para dimensionar el multiplicador del canal.
    _KC_MULT_QUANTILE = 0.8

    @staticmethod
    def signal(
        df: pd.DataFrame,
        ema_n: int = 20,
        atr_n: int = 14,
        mult: float = 1.0,
        volume_factor: float = 1.5,
    ) -> tuple[pd.Series, pd.Series]:
        """Vectorised breakout/ATR entry and exit signals.

        Parameters
        ----------
        df:
            OHLCV data frame. ``high``, ``low``, ``close`` and optionally
            ``volume`` are used to compute indicators in a single pass.
        ema_n:
            Window for the EMA forming the channel mid-line.
        atr_n:
            Lookback window for the ATR.
        mult:
            Channel width multiplier applied to the ATR.
        volume_factor:
            Minimum multiple of the 20-bar average volume required for signals.

        Returns
        -------
        tuple of pandas.Series
            ``(entries, exits)`` boolean Series suitable for
            :func:`vectorbt.Portfolio.from_signals`.
        """

        atr_series = atr(df, atr_n)
        ema = df["close"].ewm(span=ema_n, adjust=False).mean()
        upper = ema + mult * atr_series
        lower = ema - mult * atr_series
        entries = df["close"] > upper
        exits = df["close"] < lower
        if "volume" in df and volume_factor > 0:
            avg_vol = df["volume"].rolling(20).mean()
            vol_filter = df["volume"] > volume_factor * avg_vol
            entries &= vol_filter
            exits &= vol_filter
        return entries.fillna(False), exits.fillna(False)

    def __init__(
        self,
        ema_n: int = 20,
        atr_n: int = 14,
        vol_quantile: float = 0.2,
        offset_frac: float = 0.02,
        *,
        config_path: str | None = None,
        **kwargs,
    ):
        params = {**load_params(config_path), **kwargs}
        self.risk_service = params.pop("risk_service", None)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        # Valores base parametrizables para 1m.
        self.base_vol_quantile = float(params.get("vol_quantile", vol_quantile))
        self.base_offset_frac = float(params.get("offset_frac", offset_frac))
        self.volume_factor = float(params.get("volume_factor", 1.5))
        tf = str(params.get("timeframe", "3m"))
        self.timeframe = tf
        tf_minutes = timeframe_to_minutes(tf)
        cooldown_param = params.get("cooldown_bars")
        if cooldown_param is None:
            cooldown_param = 3.0
        self._cooldown_minutes = float(cooldown_param)
        self.cooldown_bars = self._cooldown_for(tf_minutes)
        self._cooldown = 0
        self._last_rpnl = 0.0
        # ``mult`` se calcula dinámicamente en ``on_bar``.
        self.mult = 1.0
        self._rq = RollingQuantileCache()

    @staticmethod
    def _tf_multiplier(tf: str | None) -> float:
        return timeframe_to_minutes(tf)

    def _cooldown_for(self, tf_minutes: float) -> int:
        if self._cooldown_minutes <= 0:
            return 0
        return max(1, int(math.ceil(self._cooldown_minutes / max(tf_minutes, 1e-9))))

    def _offset_fraction(self, tf_mult: float) -> float:
        """Return the ATR fraction used to cross the market with limit orders."""

        base = self.base_offset_frac
        if tf_mult <= 1:
            return base
        if tf_mult <= 3:
            return base * 0.75
        scale = max(tf_mult ** 0.5, 1.5)
        return base * min(scale, 6.0)

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        tf_val = bar.get("timeframe", self.timeframe)
        tf_mult = self._tf_multiplier(tf_val)
        self.cooldown_bars = self._cooldown_for(tf_mult)

        # Ajusta parámetros según el timeframe
        if tf_mult <= 3:
            ema_n = 15
            atr_n = 10
            stop_mult = 1.5
            max_hold = 10
        elif tf_mult >= 30:
            ema_n = max(self.ema_n, 30)
            atr_n = max(self.atr_n, 20)
            stop_mult = 2.0
            max_hold = 20
        else:
            ema_n = self.ema_n
            atr_n = self.atr_n
            stop_mult = 1.5
            max_hold = 20

        if len(df) < max(ema_n, atr_n) + 2:
            return None

        if self.risk_service is not None:
            rpnl = getattr(self.risk_service.pos, "realized_pnl", 0.0)
            if rpnl < self._last_rpnl and abs(self.risk_service.pos.qty) < 1e-9:
                self._cooldown = self.cooldown_bars
            self._last_rpnl = rpnl
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        atr_series = atr(df, atr_n).dropna()
        if len(atr_series) < atr_n:
            return None

        last_close = float(df["close"].iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        atr_bps = atr_val / abs(last_close) * 10000 if last_close else 0.0
        # Expose current ATR so runners and the RiskManager can adapt sizing
        bar["atr"] = atr_val

        if tf_mult <= 1:
            vol_q = 0.3
        elif tf_mult <= 3:
            vol_q = 0.25
        else:
            vol_q = max(1.0 / tf_mult, 0.05)

        window = min(len(atr_series), self.atr_n * 5)
        symbol = bar.get("symbol", "")
        if window >= self.atr_n * 2:
            rq_atr = self._rq.get(
                symbol,
                "atr",
                window=self.atr_n * 5,
                q=vol_q,
                min_periods=self.atr_n * 2,
            )
            atr_quant = float(rq_atr.update(atr_val))
            atr_bps_series = atr_series / df["close"].abs().loc[atr_series.index] * 10000
            rq_bps = self._rq.get(
                symbol,
                "atr_bps",
                window=self.atr_n * 5,
                q=vol_q,
                min_periods=self.atr_n * 2,
            )
            atr_bps_quant = float(rq_bps.update(atr_bps))
            if atr_val < atr_quant or atr_bps < atr_bps_quant:
                return None

            rq_mult = self._rq.get(
                symbol,
                "atr_mult",
                window=self.atr_n * 5,
                q=self._KC_MULT_QUANTILE,
                min_periods=self.atr_n * 2,
            )
            mult_quant = float(rq_mult.update(atr_val))
            if not math.isfinite(mult_quant) or not atr_val:
                self.mult = 1.0
            else:
                self.mult = mult_quant / atr_val

            rq_target = self._rq.get(
                symbol,
                "atr_median",
                window=self.atr_n * 5,
                q=0.5,
                min_periods=self.atr_n * 2,
            )
            target_vol = float(rq_target.update(atr_val))
            bar["target_volatility"] = target_vol
        else:
            self.mult = 1.0
            target_vol = atr_val
            bar["target_volatility"] = target_vol

        upper, lower = keltner_channels(df, ema_n, atr_n, self.mult)

        side: str | None = None
        if last_close > upper.iloc[-1]:
            side = "buy"
        elif last_close < lower.iloc[-1]:
            side = "sell"
        if side is None:
            return None

        # Filtro de volumen
        if "volume" in df:
            vol_series = df["volume"]
            avg_vol = vol_series.iloc[-20:].mean()
            if vol_series.iloc[-1] <= self.volume_factor * avg_vol:
                return None
        strength = 1.0
        sig = Signal(side, strength)
        level = float(upper.iloc[-1]) if side == "buy" else float(lower.iloc[-1])
        offset = atr_val * self._offset_fraction(tf_mult)
        if side == "buy":
            sig.limit_price = max(last_close, level + offset)
        else:
            sig.limit_price = min(last_close, level - offset)

        symbol = bar.get("symbol")
        if symbol:
            if not hasattr(self, "_last_atr"):
                self._last_atr: dict[str, float] = {}
            self._last_atr[symbol] = atr_val
            if not hasattr(self, "_last_target_vol"):
                self._last_target_vol: dict[str, float] = {}
            self._last_target_vol[symbol] = target_vol

        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(strength, last_close)
            stop = self.risk_service.initial_stop(
                last_close, side, atr_val, atr_mult=stop_mult
            )
            self.trade = {
                "side": side,
                "entry_price": last_close,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "bars_held": 0,
                "max_hold": max_hold,
            }

        return self.finalize_signal(bar, last_close, sig)
