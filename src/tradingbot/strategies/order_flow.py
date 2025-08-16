import pandas as pd
from .base import Strategy, Signal
from ..data.features import order_flow_imbalance


class OrderFlow(Strategy):
    """Estrategia basada en el flujo de órdenes (Order Flow Imbalance).

    Agrega el OFI en una ventana configurable y genera señales cuando la
    suma supera umbrales superiores o inferiores.
    """

    name = "order_flow"

    def __init__(self, window: int = 5, upper: float = 0.0, lower: float = 0.0):
        self.window = window
        self.upper = upper
        self.lower = lower

    def on_bar(self, bar: dict) -> Signal | None:
        """Procesa una nueva barra y devuelve una señal basada en OFI."""
        df: pd.DataFrame = bar["window"]
        if not {"bid_qty", "ask_qty"}.issubset(df.columns):
            return None
        if len(df) <= self.window:
            return None

        ofi = order_flow_imbalance(df[["bid_qty", "ask_qty"]])
        agg = ofi.rolling(self.window).sum().iloc[-1]

        if agg > self.upper:
            return Signal("buy", 1.0)
        if agg < self.lower:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
