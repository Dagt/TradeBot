import pytest

from tradingbot.risk.manager import RiskManager
from tradingbot.strategies.base import Strategy, Signal, record_signal_metrics


class DummyStrategy(Strategy):
    name = "dummy"

    def __init__(self, risk):
        self.risk = risk

    @record_signal_metrics
    def on_bar(self, bar):
        return Signal("buy")


def test_risk_vol_sizing(synthetic_volatility):
    rm = RiskManager(max_pos=10, vol_target=0.02)
    strat = DummyStrategy(rm)
    bar = {"volatility": synthetic_volatility}
    sig = strat.on_bar(bar)
    expected = rm.max_pos + min(rm.max_pos, rm.max_pos * rm.vol_target / synthetic_volatility)
    assert sig.strength == pytest.approx(expected)
