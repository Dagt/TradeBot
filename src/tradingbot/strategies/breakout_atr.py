import pandas as pd
from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import atr, keltner_channels

PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "mult": "Multiplicador aplicado al ATR",
    # ``min_atr`` se interpreta en unidades de precio y ya no depende del
    # tamaño de la vela.
    "min_atr": "ATR mínimo para operar (independiente del timeframe)",
    # Umbral relativo expresado en puntos básicos del precio.
    "min_atr_bps": "ATR mínimo relativo en bps",
    "config_path": "Ruta opcional al archivo de configuración",
}


class BreakoutATR(Strategy):
    name = "breakout_atr"

    def __init__(
        self,
        ema_n: int = 20,
        atr_n: int = 14,
        mult: float = 1.0,
        min_atr: float = 0.0,
        min_atr_bps: float = 0.0,
        *,
        config_path: str | None = None,
        **kwargs,
    ):
        params = {**load_params(config_path), **kwargs}
        self.risk_service = params.pop("risk_service", None)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        self.mult = float(params.get("mult", mult))
        self.min_atr = float(params.get("min_atr", min_atr))
        self.min_atr_bps = float(params.get("min_atr_bps", min_atr_bps))

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = float(df["close"].iloc[-1])
        atr_val = float(atr(df, self.atr_n).iloc[-1])
        atr_bps = atr_val / abs(last_close) * 10000 if last_close else 0.0

        if atr_val < self.min_atr or atr_bps < self.min_atr_bps:
            return None

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
        sig.limit_price = level

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
