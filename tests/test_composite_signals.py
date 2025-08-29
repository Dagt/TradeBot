import pytest
import pytest

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


def test_tp_sl_applied_on_consensus():
    cs = CompositeSignals(
        [(BuyStrat, {}), (BuyStrat, {}), (SellStrat, {})],
        tp_pct=0.05,
        sl_pct=0.02,
    )
    sig = cs.on_bar({"close": 100})
    assert sig is not None
    assert sig.side == "buy"
    assert pytest.approx(sig.take_profit, rel=1e-9) == 105.0
    assert pytest.approx(sig.stop_loss, rel=1e-9) == 98.0
