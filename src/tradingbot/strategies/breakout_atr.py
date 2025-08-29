import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import keltner_channels


class BreakoutATR(Strategy):
    name = "breakout_atr"

    def __init__(
        self,
        ema_n: int = 20,
        atr_n: int = 14,
        mult: float = 1.0,
        min_bars_between_trades: int = 1,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
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
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

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

        if self.pos_side == 0:
            sig: Signal | None = None
            if last_close > upper.iloc[-1]:
                sig = Signal("buy", 1.0)
                self.pos_side = 1
                self.entry_price = last_close
                self.hold_bars = 0
            elif last_close < lower.iloc[-1]:
                sig = Signal("sell", 1.0)
                self.pos_side = -1
                self.entry_price = last_close
                self.hold_bars = 0
            if sig and sig.side in {"buy", "sell"}:
                if (
                    self._last_trade_idx is not None
                    and self._last_trade_side is not None
                    and sig.side != self._last_trade_side
                    and current_idx - self._last_trade_idx
                    < self.min_bars_between_trades
                ):
                    self.pos_side = 0
                    self.entry_price = None
                    return None
                self._last_trade_idx = current_idx
                self._last_trade_side = sig.side
                return sig
            return None

        # manage existing position
        self.hold_bars += 1
        assert self.entry_price is not None
        pnl_bps = (
            (last_close - self.entry_price) / self.entry_price * 10000 * self.pos_side
        )
        if (
            pnl_bps >= self.tp_bps
            or pnl_bps <= -self.sl_bps
            or self.hold_bars >= self.max_hold_bars
        ):
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            self._last_trade_idx = current_idx
            self._last_trade_side = side
            return Signal(side, 1.0)
        return None
