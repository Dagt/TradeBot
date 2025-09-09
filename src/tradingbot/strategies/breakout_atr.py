import re
import pandas as pd
from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import atr, keltner_channels, calc_ofi
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager

PARAM_INFO = {
    "ema_n": "Periodo de la EMA para la línea central",
    "atr_n": "Periodo del ATR usado en los canales",
    "vol_quantile": "Percentil base para filtrar baja volatilidad (1m)",
    "offset_frac": "Fracción base del ATR usada como offset (1m)",
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
    ATR y lo incrementan de forma progresiva si la orden expira sin
    ejecutarse, buscando mejorar la tasa de ejecución sin requerir
    parámetros adicionales.
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
        tf = params.get("timeframe", "3m")
        self.cooldown_bars = int(params.get("cooldown_bars", 3 if tf in {"1m", "3m"} else 0))
        self._cooldown = 0
        self._last_rpnl = 0.0
        self.timeframe = tf
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

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        tf_mult = self._tf_multiplier(bar.get("timeframe"))

        if len(df) < 2:
            return None
        last_close = float(df["close"].iloc[-1])

        # --- Basic microstructure guard
        spread_bps = 0.0
        if {"bid", "ask"}.issubset(df.columns):
            spread = float(df["ask"].iloc[-1] - df["bid"].iloc[-1])
            spread_bps = spread / last_close * 10000 if last_close else 0.0
        else:
            high = float(df["high"].iloc[-1]) if "high" in df.columns else last_close
            low = float(df["low"].iloc[-1]) if "low" in df.columns else last_close
            spread_bps = (high - low) / last_close * 10000 if last_close else 0.0
        bar["spread_bps"] = spread_bps
        if spread_bps > 5:
            return None

        # --- Manage existing position: partials and time-stop
        trade = getattr(self, "trade", None)
        if trade:
            trade["bars_held"] = trade.get("bars_held", 0) + 1
            take1 = trade.get("take1")
            if take1 is None:
                take1 = (
                    trade["entry_price"] + 0.5 * trade["atr"]
                    if trade["side"] == "buy"
                    else trade["entry_price"] - 0.5 * trade["atr"]
                )
                trade["take1"] = take1
            if not trade.get("tp_hit") and (
                (trade["side"] == "buy" and last_close >= take1)
                or (trade["side"] == "sell" and last_close <= take1)
            ):
                trade["tp_hit"] = True
                trade["qty"] = float(trade.get("qty", 0)) * 0.5
                sig = Signal(trade["side"], 0.5, reduce_only=True)
                return self.finalize_signal(bar, last_close, sig)
            if trade["bars_held"] >= 6 and not trade.get("tp_hit"):
                side = "sell" if trade["side"] == "buy" else "buy"
                sig = Signal(side, 1.0, reduce_only=True)
                self.trade = None
                return self.finalize_signal(bar, last_close, sig)

        # Ajusta parámetros según el timeframe
        if tf_mult <= 3:
            ema_n = 15
            atr_n = 10
        elif tf_mult >= 30:
            ema_n = max(self.ema_n, 30)
            atr_n = max(self.atr_n, 20)
        else:
            ema_n = self.ema_n
            atr_n = self.atr_n

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
        vol_bps = atr_val / abs(last_close) * 10000 if last_close else 0.0
        bar["atr"] = atr_val
        bar["vol_bps"] = vol_bps
        if vol_bps < 8:
            return None

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
            rq_bps = self._rq.get(
                symbol,
                "atr_bps",
                window=self.atr_n * 5,
                q=vol_q,
                min_periods=self.atr_n * 2,
            )
            vol_bps_quant = float(rq_bps.update(vol_bps))
            if atr_val < atr_quant or vol_bps < vol_bps_quant:
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

        tf = getattr(self, "timeframe", "")
        k_mult = 1.0
        if tf in ("1m", "3m", "5m"):
            k_mult = 1.3
        upper, lower = keltner_channels(df, ema_n, atr_n, self.mult * k_mult)

        prev_close = float(df["close"].iloc[-2])
        bull_break = (last_close > upper.iloc[-1]) and (prev_close <= upper.iloc[-2])
        bear_break = (last_close < lower.iloc[-1]) and (prev_close >= lower.iloc[-2])
        if not (bull_break or bear_break):
            return None
        side = "buy" if bull_break else "sell"

        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        vol_spike = True
        if "volume" in df:
            vol_spike = (
                df["volume"].iloc[-1]
                > df["volume"].rolling(20).median().iloc[-2] * 1.5
            )
        if side == "buy" and (ofi_val < 0 or not vol_spike):
            return None
        if side == "sell" and (ofi_val > 0 or not vol_spike):
            return None
        strength = 1.0
        sig = Signal(side, strength)
        level = float(upper.iloc[-1]) if side == "buy" else float(lower.iloc[-1])
        if tf_mult <= 1:
            offset = atr_val * 0.05
        elif tf_mult <= 3:
            offset = atr_val * 0.04
        else:
            offset = atr_val * 0.02 * tf_mult
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
            stop = self.risk_service.initial_stop(
                last_close, side, atr_val, atr_mult=1.5
            )
            self.trade = {
                "side": side,
                "entry_price": last_close,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "bars_held": 0,
            }
        else:
            self.trade = {
                "side": side,
                "entry_price": last_close,
                "qty": strength,
                "stop": last_close - 1.5 * atr_val
                if side == "buy"
                else last_close + 1.5 * atr_val,
                "atr": atr_val,
                "bars_held": 0,
            }

        return self.finalize_signal(bar, last_close, sig)
