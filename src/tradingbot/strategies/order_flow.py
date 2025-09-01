import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import calc_ofi


PARAM_INFO = {
    "window": "Ventana para promediar el OFI",
    "buy_threshold": "Umbral de compra para OFI",
    "sell_threshold": "Umbral de venta para OFI",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class OrderFlow(Strategy):
    """Order Flow Imbalance strategy.

    Calculates the mean Order Flow Imbalance (OFI) over a rolling window and
    issues buy/sell signals when the mean exceeds the configured thresholds.
    """

    name = "order_flow"

    def __init__(
        self,
        window: int = 3,
        buy_threshold: float = 1.0,
        sell_threshold: float = 1.0,
        min_volatility: float = 0.0,
        **kwargs,
    ):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_volatility = min_volatility
        self.risk_service = kwargs.get("risk_service")
        self.trade: dict | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None
        price = bar.get("close")
        if price is None:
            col = (
                "close"
                if "close" in df.columns
                else "price" if "price" in df.columns else None
            )
            if col is None:
                return None
            price = float(df[col].iloc[-1])
        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, price)
            trade_state = {**self.trade, "current_price": price}
            decision = self.risk_service.manage_position(trade_state)
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                sig = Signal(side, 1.0)
                sig.limit_price = price
                return sig
            if decision in {"scale_in", "scale_out"}:
                self.trade["strength"] = trade_state.get("strength", 1.0)
                sig = Signal(self.trade["side"], self.trade["strength"])
                sig.limit_price = price
                return sig
            return None

        vol_bps = float("inf")
        price_col = "close" if "close" in df.columns else None
        if price_col:
            closes = df[price_col]
            returns = closes.pct_change().dropna()
            vol = (
                returns.rolling(self.window).std().iloc[-1]
                if len(returns) >= self.window
                else 0.0
            )
            vol_bps = vol * 10000
        if vol_bps < self.min_volatility:
            return None

        ofi_series = calc_ofi(df[list(needed)])
        ofi_mean = ofi_series.iloc[-self.window :].mean()
        if ofi_mean > self.buy_threshold:
            side = "buy"
        elif ofi_mean < -self.sell_threshold:
            side = "sell"
        else:
            return None
        strength = 1.0
        if self.risk_service:
            qty = self.risk_service.calc_position_size(strength, price)
            trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "strength": strength,
            }
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(price, side, atr)
            if atr is not None:
                trade["atr"] = atr
            self.risk_service.update_trailing(trade, price)
            self.trade = trade
        if price is None:
            return None
        sig = Signal(side, strength)
        sig.limit_price = price
        return sig
