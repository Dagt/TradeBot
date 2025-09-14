import logging
from tradingbot.strategies.base import Strategy, Signal

class DummyRisk:
    def __init__(self):
        self.min_order_qty = 1.0
        self.min_notional = 10.0
    def get_trade(self, symbol):
        return None
    def calc_position_size(self, strength, price, **kwargs):
        return strength

class SmallStrat(Strategy):
    name = "small"
    def on_bar(self, bar):
        sig = Signal("buy", strength=0.5)
        return self.finalize_signal(bar, bar["price"], sig)


def test_subminimum_signal_is_discarded(caplog):
    risk = DummyRisk()
    strat = SmallStrat()
    strat.risk_service = risk
    bar = {"price": 1.0, "symbol": "BTC"}
    with caplog.at_level(logging.INFO):
        sig = strat.on_bar(bar)
    assert sig is None
    assert "orden submínima" in caplog.text
