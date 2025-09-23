import logging

from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
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


class StrengthSequenceStrat(Strategy):
    name = "strength_sequence"

    def __init__(self, strengths: list[float]):
        self._strengths = strengths
        self._i = 0

    def on_bar(self, bar):
        if self._i >= len(self._strengths):
            return None
        strength = self._strengths[self._i]
        self._i += 1
        sig = Signal("buy", strength=strength)
        return self.finalize_signal(bar, bar["price"], sig)


def test_calc_position_size_uses_updated_strength(monkeypatch):
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    account = Account(float("inf"), cash=1000.0)
    risk = RiskService(guard, account=account, risk_per_trade=1.0)

    strengths_seen: list[float] = []
    original_calc = RiskService.calc_position_size

    def _record(self, strength: float, price: float, **kwargs):
        strengths_seen.append(strength)
        return original_calc(self, strength, price, **kwargs)

    monkeypatch.setattr(RiskService, "calc_position_size", _record)

    strat = StrengthSequenceStrat([0.1, 0.75])
    strat.risk_service = risk
    bar = {"price": 100.0, "symbol": "BTC"}

    first = strat.on_bar(bar)
    second = strat.on_bar(bar)

    assert first is not None
    assert second is not None
    assert strengths_seen == [0.1, 0.75]
