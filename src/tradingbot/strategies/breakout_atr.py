import pandas as pd
from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import atr, keltner_channels

PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "config_path": "Ruta opcional al archivo de configuración",
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

    # Percentil utilizado para estimar el umbral de volatilidad.
    _VOL_QUANTILE = 0.2
    # Percentil usado para dimensionar el multiplicador del canal.
    _KC_MULT_QUANTILE = 0.8

    def __init__(
        self,
        ema_n: int = 20,
        atr_n: int = 14,
        *,
        config_path: str | None = None,
        **kwargs,
    ):
        params = {**load_params(config_path), **kwargs}
        self.risk_service = params.pop("risk_service", None)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        # ``mult`` se calcula dinámicamente en ``on_bar``.
        self.mult = 1.0

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

        window = min(len(atr_series), self.atr_n * 5)
        if window >= self.atr_n * 2:
            atr_quant = float(
                atr_series.rolling(window).quantile(self._VOL_QUANTILE).iloc[-1]
            )
            atr_bps_series = (
                atr_series / df["close"].abs().loc[atr_series.index] * 10000
            )
            atr_bps_quant = float(
                atr_bps_series.rolling(window).quantile(self._VOL_QUANTILE).iloc[-1]
            )
            if atr_val < atr_quant or atr_bps < atr_bps_quant:
                return None

            # Multiplicador dinámico basado en el percentil reciente del ATR.
            mult_quant = float(
                atr_series.rolling(window).quantile(self._KC_MULT_QUANTILE).iloc[-1]
            )
            self.mult = mult_quant / atr_val if atr_val else 1.0
            # Target de volatilidad para ajustes de riesgo
            target_vol = float(atr_series.rolling(window).median().iloc[-1])
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
        offset = 0.1 * atr_val
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
