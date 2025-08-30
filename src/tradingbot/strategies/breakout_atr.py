import pandas as pd
from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import atr, keltner_channels

PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "mult": "Multiplicador aplicado al ATR",
    "min_atr": "ATR mínimo para operar",
    "min_volatility": "Volatilidad mínima reciente en bps",
    "min_edge_bps": "Edge mínimo en puntos básicos para operar",
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
        min_volatility: float = 0.0,
        min_edge_bps: float = 0.0,
        *,
        config_path: str | None = None,
    ):
        params = load_params(config_path)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        self.mult = float(params.get("mult", mult))
        self.min_atr = float(params.get("min_atr", min_atr))
        self.min_volatility = float(params.get("min_volatility", min_volatility))
        self.min_edge_bps = float(params.get("min_edge_bps", min_edge_bps))

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = df["close"].iloc[-1]
        atr_val = atr(df, self.atr_n).iloc[-1]
        atr_bps = atr_val / abs(last_close) * 10000 if last_close else 0.0

        if atr_val < self.min_atr or atr_bps < self.min_volatility:
            return None

        if last_close > upper.iloc[-1]:
            expected_edge_bps = (last_close - upper.iloc[-1]) / abs(last_close) * 10000
            if expected_edge_bps <= self.min_edge_bps:
                return None
            return Signal("buy", 1.0, expected_edge_bps=expected_edge_bps)
        if last_close < lower.iloc[-1]:
            expected_edge_bps = (lower.iloc[-1] - last_close) / abs(last_close) * 10000
            if expected_edge_bps <= self.min_edge_bps:
                return None
            return Signal("sell", 1.0, expected_edge_bps=expected_edge_bps)
        return None
