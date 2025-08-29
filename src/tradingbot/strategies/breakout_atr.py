import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import atr, keltner_channels


PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "mult": "Multiplicador aplicado al ATR",
    "min_bars_between_trades": "Barras mínimas entre operaciones",
    "tp_bps": "Take profit en puntos básicos",
    "sl_bps": "Stop loss en puntos básicos",
    "max_hold_bars": "Máximo de barras en posición",
    "min_atr": "ATR mínimo para operar",
    "trail_atr_mult": "Multiplicador del trailing stop basado en ATR",
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
        min_bars_between_trades: int = 1,
        tp_bps: float = 5.0,
        sl_bps: float = 5.0,
        max_hold_bars: int = 3,
        min_atr: float = 0.0,
        trail_atr_mult: float = 1.0,
        min_edge_bps: float = 0.0,
        *,
        config_path: str | None = None,
    ):
        params = load_params(config_path)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        self.mult = float(params.get("mult", mult))
        mbbt = params.get("min_bars_between_trades", min_bars_between_trades)
        self.min_bars_between_trades = max(int(mbbt), 1)
        self._last_trade_idx: int | None = None
        self._last_trade_side: str | None = None
        self.tp_bps = float(params.get("tp_bps", tp_bps))
        self.sl_bps = float(params.get("sl_bps", sl_bps))
        self.max_hold_bars = int(params.get("max_hold_bars", max_hold_bars))
        self.min_atr = float(params.get("min_atr", min_atr))
        self.trail_atr_mult = float(params.get("trail_atr_mult", trail_atr_mult))
        self.min_edge_bps = float(params.get("min_edge_bps", min_edge_bps))
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0
        self.trailing_stop: float | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        # bar should include a small rolling window (as dict of lists)
        # or a pandas row with context
        df: pd.DataFrame = bar["window"]
        # expects columns: open, high, low, close, volume
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = df["close"].iloc[-1]
        current_idx = len(df) - 1
        atr_val = atr(df, self.atr_n).iloc[-1]

        if self.pos_side == 0:
            if atr_val < self.min_atr:
                return None
            side: str | None = None
            expected_edge_bps = 0.0
            trail_stop: float | None = None
            if last_close > upper.iloc[-1]:
                expected_edge_bps = (
                    (last_close - upper.iloc[-1]) / abs(last_close) * 10000
                )
                side = "buy"
                trail_stop = last_close - atr_val * self.trail_atr_mult
            elif last_close < lower.iloc[-1]:
                expected_edge_bps = (
                    (lower.iloc[-1] - last_close) / abs(last_close) * 10000
                )
                side = "sell"
                trail_stop = last_close + atr_val * self.trail_atr_mult
            if side is None or expected_edge_bps <= self.min_edge_bps:
                return None
            if (
                self._last_trade_idx is not None
                and self._last_trade_side is not None
                and side != self._last_trade_side
                and current_idx - self._last_trade_idx < self.min_bars_between_trades
            ):
                return None
            self.pos_side = 1 if side == "buy" else -1
            self.entry_price = last_close
            self.hold_bars = 0
            self.trailing_stop = trail_stop
            self._last_trade_idx = current_idx
            self._last_trade_side = side
            return Signal(side, 1.0, expected_edge_bps=expected_edge_bps)
        

        # manage existing position
        self.hold_bars += 1
        assert self.entry_price is not None and self.trailing_stop is not None
        pnl_bps = (
            (last_close - self.entry_price) / self.entry_price * 10000 * self.pos_side
        )
        if self.pos_side > 0:
            self.trailing_stop = max(
                self.trailing_stop, last_close - atr_val * self.trail_atr_mult
            )
            stop_hit = last_close <= self.trailing_stop
        else:
            self.trailing_stop = min(
                self.trailing_stop, last_close + atr_val * self.trail_atr_mult
            )
            stop_hit = last_close >= self.trailing_stop
        if (
            pnl_bps >= self.tp_bps
            or pnl_bps <= -self.sl_bps
            or self.hold_bars >= self.max_hold_bars
            or stop_hit
        ):
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            self.trailing_stop = None
            self._last_trade_idx = current_idx
            self._last_trade_side = side
            return Signal(side, 1.0)
        return None
