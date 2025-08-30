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
        risk_service=None,
        config_path: str | None = None,
    ):
        params = load_params(config_path)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        self.mult = float(params.get("mult", mult))
        self.min_atr = float(params.get("min_atr", min_atr))
        self.min_volatility = float(params.get("min_volatility", min_volatility))
        self.min_edge_bps = float(params.get("min_edge_bps", min_edge_bps))
        self.risk_service = risk_service
        self.trade: dict | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = float(df["close"].iloc[-1])
        if self.trade and self.risk_service:
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
        atr_val = float(atr(df, self.atr_n).iloc[-1])
        atr_bps = atr_val / abs(last_close) * 10000 if last_close else 0.0

        if atr_val < self.min_atr or atr_bps < self.min_volatility:
            return None

        side: str | None = None
        if last_close > upper.iloc[-1]:
            edge_bps = (last_close - upper.iloc[-1]) / abs(last_close) * 10000
            if edge_bps <= self.min_edge_bps:
                return None
            side = "buy"
        elif last_close < lower.iloc[-1]:
            edge_bps = (lower.iloc[-1] - last_close) / abs(last_close) * 10000
            if edge_bps <= self.min_edge_bps:
                return None
            side = "sell"
        if side is None:
            return None
        strength = 1.0
        if self.risk_service:
            qty = self.risk_service.calc_position_size(strength, last_close)
            trade = {"side": side, "entry_price": last_close, "qty": qty, "strength": strength}
            trade["stop"] = self.risk_service.initial_stop(
                last_close, side, atr_val
            )
            trade["atr"] = atr_val
            self.risk_service.update_trailing(trade, last_close)
            self.trade = trade
        return Signal(side, strength)
