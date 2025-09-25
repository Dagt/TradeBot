import pandas as pd
import math

from .base import Strategy, Signal, record_signal_metrics, timeframe_to_minutes
from ..data.features import rsi
from ..filters.liquidity import LiquidityFilterManager

liquidity = LiquidityFilterManager()

PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "trend_ma": "Ventana para la media móvil de tendencia",
    "trend_rsi_n": "Ventana del RSI para medir tendencia",
    "trend_threshold": "Umbral para considerar la tendencia fuerte",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class MeanReversion(Strategy):
    """RSI based mean reversion strategy with adaptive strength.

    Generates ``buy`` or ``sell`` signals when the RSI deviates from
    dynamically determined thresholds. Signal strength scales with the
    distance from the threshold.
    """

    name = "mean_reversion"
    max_signal_strength = 2.5

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)

        tf = str(kwargs.get("timeframe", "1m"))
        self.timeframe = tf
        tf_minutes = timeframe_to_minutes(tf)
        self._base_timeframe_minutes = tf_minutes

        trend_ma_min = kwargs.get("trend_ma", 50)
        self.trend_ma = max(1, int(trend_ma_min / tf_minutes))

        trend_rsi_min = kwargs.get("trend_rsi_n", 50)
        self.trend_rsi_n = max(1, int(trend_rsi_min / tf_minutes))

        thresh = kwargs.get("trend_threshold", 10.0)
        self.trend_threshold = float(thresh) / tf_minutes
        if tf in {"30m", "1h"}:
            self.trend_threshold *= 1.5

        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.only_buy_dip = kwargs.get("only_buy_dip", tf in {"30m", "1h"})
        default_time_stop_bars = 0.0
        min_time_stop_bars = 1
        if tf_minutes >= 30.0:
            if tf_minutes < 60.0:
                min_time_stop_bars = 5
            else:
                min_time_stop_bars = 3
            default_time_stop_bars = float(min_time_stop_bars)

        self._time_stop_target_bars = 0
        self._min_time_stop_bars = 0
        time_stop_param = float(kwargs.get("time_stop", default_time_stop_bars))
        if time_stop_param > 0:
            target_bars = int(math.ceil(time_stop_param))
            target_bars = max(min_time_stop_bars, target_bars)
            self._time_stop_target_bars = target_bars
            self._min_time_stop_bars = min_time_stop_bars

        self._time_stop_minutes = self._time_stop_target_bars * tf_minutes if self._time_stop_target_bars else 0
        self.time_stop = 0
        self._open_bars: dict[str, int] = {}
        self.risk_service = kwargs.get("risk_service")

    def auto_threshold(self, rsi_series: pd.Series) -> tuple[float, float]:
        """Derive upper and lower RSI bounds from recent variability."""

        dev = rsi_series.rolling(self.rsi_n).std().iloc[-1]
        dev = 10.0 if pd.isna(dev) else float(dev)
        upper = 50 + dev
        lower = 50 - dev
        return upper, lower

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        price_series = df[price_col]
        price = float(price_series.iloc[-1])
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]

        bar_timeframe = bar.get("timeframe", self.timeframe)
        tf_minutes = timeframe_to_minutes(bar_timeframe)
        base_minutes = self._base_timeframe_minutes
        if self._time_stop_target_bars <= 0 or tf_minutes <= 0:
            time_stop_bars = 0
        else:
            desired_minutes = self._time_stop_target_bars * base_minutes
            scaled_bars = int(math.ceil(desired_minutes / tf_minutes))
            min_bars = self._min_time_stop_bars or 1
            time_stop_bars = max(min_bars, scaled_bars)
        self.time_stop = time_stop_bars

        if time_stop_bars and self.risk_service is not None and bar.get("symbol"):
            sym = bar["symbol"]
            trade = self.risk_service.get_trade(sym)
            if trade:
                cnt = self._open_bars.get(sym, 0) + 1
                self._open_bars[sym] = cnt
                if cnt >= time_stop_bars:
                    side_exit = "sell" if trade.get("side") == "buy" else "buy"
                    return self.finalize_signal(bar, price, Signal(side_exit, 1.0))
            else:
                self._open_bars[sym] = 0

        returns = price_series.pct_change().dropna()
        vol_series = (
            returns.rolling(self.rsi_n).std()
            if len(returns) >= self.rsi_n
            else pd.Series(dtype=float)
        )
        vol = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
        if vol * 10000 < self.min_volatility:
            return None
        abs_price = max(abs(price), 1e-9)
        price_vol = abs_price * vol if math.isfinite(vol) and vol > 0 else 0.0
        bar["volatility"] = price_vol
        target_vol = price_vol
        if len(vol_series.dropna()):
            target_candidate = float(vol_series.dropna().median())
            if math.isfinite(target_candidate) and target_candidate > 0:
                target_vol = abs_price * max(target_candidate, vol)
        bar["target_volatility"] = target_vol

        high = df.get("high")
        low = df.get("low")
        atr_val = 0.0
        if high is not None and low is not None:
            prev_close = price_series.shift()
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr_series = tr.rolling(self.rsi_n).mean().dropna()
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0

        trend_dir = 0
        if len(df) >= self.trend_ma:
            ma = price_series.rolling(self.trend_ma).mean().iloc[-1]
            if not pd.isna(ma) and ma != 0:
                diff_pct = (price - ma) / ma * 100
                if diff_pct > self.trend_threshold:
                    trend_dir = 1
                elif diff_pct < -self.trend_threshold:
                    trend_dir = -1
        elif len(df) >= self.trend_rsi_n:
            trsi = rsi(df, self.trend_rsi_n).iloc[-1]
            if trsi > 50 + self.trend_threshold:
                trend_dir = 1
            elif trsi < 50 - self.trend_threshold:
                trend_dir = -1

        upper, lower = self.auto_threshold(rsi_series)
        if trend_dir == 1:
            upper += self.trend_threshold
        elif trend_dir == -1:
            lower -= self.trend_threshold

        prev_rsi = rsi_series.iloc[-2]
        prev_price = float(price_series.iloc[-2])

        if last_rsi > upper:
            if self.timeframe in {"5m", "15m"} and not (
                prev_rsi > last_rsi and price < prev_price
            ):
                return self.finalize_signal(bar, price, None)
            strength = max(0.3, min(2.5, (last_rsi - upper) / max(1.0, 100 - upper) * 3.0))
            side = "sell"
        elif last_rsi < lower:
            if self.timeframe in {"5m", "15m"} and not (
                prev_rsi < last_rsi and price > prev_price
            ):
                return self.finalize_signal(bar, price, None)
            strength = max(0.3, min(2.5, (lower - last_rsi) / max(1.0, lower) * 3.0))
            side = "buy"
        else:
            return self.finalize_signal(bar, price, None)

        if side == "sell" and trend_dir == 1 and self.only_buy_dip:
            return self.finalize_signal(bar, price, None)

        sig = Signal(side, strength)
        offset = max(atr_val * 0.5, price * 0.001)
        direction = -1 if side == "sell" else 1
        base_price = price
        sig.limit_price = base_price + direction * offset
        sig.metadata.update(
            {
                "base_price": base_price,
                "limit_offset": abs(offset),
                "max_offset": abs(price * 0.005),
                "step_mult": 0.4,
                "chase": False,
                "decay": 0.7,
                "min_offset": price * 0.0005,
            }
        )
        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(
                strength,
                price,
                volatility=bar.get("volatility"),
                target_volatility=bar.get("target_volatility"),
                clamp=False,
            )
            stop = self.risk_service.initial_stop(price, side, atr_val)
            if (
                side == "sell"
                and trend_dir == 1
                and self.timeframe in {"5m", "15m"}
            ):
                stop = price + 1.5 * atr_val
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "target_volatility": bar.get("target_volatility"),
                "strength": strength,
            }

        return self.finalize_signal(bar, price, sig)


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate mean reversion signals for backtesting."""

    df = data.copy()
    window = params.get("window", 14)
    threshold = params.get("threshold", 0.0)
    position_size = params.get("position_size", 1)
    fee = params.get("fee", 0.0)
    slippage = params.get("slippage", 0.0)

    ma = df["price"].rolling(window).mean()
    df["signal"] = 0
    df.loc[df["price"] < ma - threshold, "signal"] = 1
    df.loc[df["price"] > ma + threshold, "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0) * position_size
    df["fee"] = df["position"].abs() * fee
    df["slippage"] = df["position"].abs() * slippage

    return df[["signal", "position", "fee", "slippage"]]
