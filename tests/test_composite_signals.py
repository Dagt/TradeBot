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


def test_no_consensus_returns_none():
    cs = CompositeSignals([(BuyStrat, {}), (SellStrat, {})])
    assert cs.on_bar({"close": 100}) is None


def test_consensus_returns_signal():
    cs = CompositeSignals([(BuyStrat, {}), (BuyStrat, {}), (SellStrat, {})])
    sig = cs.on_bar({"close": 100})
    assert sig is not None and sig.side == "buy"
