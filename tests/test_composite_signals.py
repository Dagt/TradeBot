import pandas as pd

from tradingbot.strategies.base import Strategy, Signal
from tradingbot.strategies.composite_signals import CompositeSignals


class BuyStrat(Strategy):
    name = "buy"

    def on_bar(self, bar):
        return Signal("buy", 1.0)


class SellStrat(Strategy):
    name = "sell"

    def on_bar(self, bar):
        return Signal("sell", 1.0)


def _bar(price: float = 100.0):
    df = pd.DataFrame({"close": [price]})
    return {"window": df, "symbol": "BTCUSDT"}


def test_no_consensus_returns_none():
    cs = CompositeSignals([(BuyStrat, {}), (SellStrat, {})])
    assert cs.on_bar(_bar()) is None


def test_consensus_returns_signal():
    cs = CompositeSignals([(BuyStrat, {}), (BuyStrat, {}), (SellStrat, {})])
    sig = cs.on_bar(_bar())
    assert sig is not None and sig.side == "buy"


def test_composite_signals_risk_closes_position():
    class DummyRisk:
        updated = False

        def get_trade(self, symbol):
            return {"symbol": symbol, "side": "buy"}

        def update_trailing(self, trade, price):
            self.updated = True

        def manage_position(self, trade, signal):
            return "close"

    rs = DummyRisk()
    cs = CompositeSignals([(BuyStrat, {}), (BuyStrat, {}), (SellStrat, {})], risk_service=rs)
    sig = cs.on_bar(_bar())
    assert rs.updated
    assert sig and sig.side == "sell" and sig.limit_price == 100.0
