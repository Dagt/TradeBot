import re
import pandas as pd
from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import atr, keltner_channels
from ..utils.rolling_quantile import RollingQuantileCache

PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "vol_quantile": "Percentil base para filtrar baja volatilidad (1m)",
    "offset_frac": "Fracción base del ATR usada como offset (1m)",
}


class BreakoutATR(Strategy):
    """Keltner breakout con filtros de volatilidad autocalibrados.

    El umbral de volatilidad y el ancho del canal se recalculan en cada
    barra usando percentiles recientes del ATR, por lo que el usuario no
    necesita ajustar parámetros adicionales para filtrar períodos de baja
    volatilidad. Las órdenes límite aplican un pequeño offset basado en el
    ATR y lo incrementan de forma progresiva si la orden expira sin
    ejecutarse, buscando mejorar la tasa de ejecución sin requerir
    parámetros adicionales.
    """

    name = "breakout_atr"

    # Percentil usado para dimensionar el multiplicador del canal.
    _KC_MULT_QUANTILE = 0.8

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
        # ``mult`` se calcula dinámicamente en ``on_bar``.
        self.mult = 1.0
        self._rq = RollingQuantileCache()

    @staticmethod
    def _tf_multiplier(tf: str | None) -> float:
        if not tf:
            return 1.0
        m = re.fullmatch(r"(\d+)([smhd])", str(tf))
        if not m:
            return 1.0
        value, unit = int(m.group(1)), m.group(2)
        factors = {"s": 1 / 60, "m": 1, "h": 60, "d": 1440}
        return value * factors.get(unit, 1.0)

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None

        atr_series = atr(df, self.atr_n).dropna()
        if len(atr_series) < self.atr_n:
            return None

        last_close = float(df["close"].iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        atr_bps = atr_val / abs(last_close) * 10000 if last_close else 0.0
        # Expose current ATR so runners and the RiskManager can adapt sizing
        bar["atr"] = atr_val

        tf_mult = self._tf_multiplier(bar.get("timeframe"))
        vol_q = max(0.0, min(1.0, self.base_vol_quantile / tf_mult))

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
            self.mult = mult_quant / atr_val if atr_val else 1.0

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

        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)

        side: str | None = None
        if last_close > upper.iloc[-1]:
            side = "buy"
        elif last_close < lower.iloc[-1]:
            side = "sell"
        if side is None:
            return None
        strength = 1.0
        sig = Signal(side, strength)
        level = float(upper.iloc[-1]) if side == "buy" else float(lower.iloc[-1])
        offset = atr_val * self.base_offset_frac * tf_mult
        sig.limit_price = level + offset if side == "buy" else level - offset

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
            stop = self.risk_service.initial_stop(last_close, side, atr_val)
            self.trade = {
                "side": side,
                "entry_price": last_close,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
            }

        return self.finalize_signal(bar, last_close, sig)
