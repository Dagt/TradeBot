import numpy as np

import pytest

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.strategies.ml_models import MLStrategy


class StubModel:
    def __init__(self, proba: float):
        self.proba = proba

    def predict_proba(self, X):
        return np.array([[1 - self.proba, self.proba]])


def test_ml_strategy_margin_and_entry():
    stub = StubModel(0.55)
    strat = MLStrategy(model=stub, margin=0.1)
    strat.scaler.fit([[0.0]])
    bar = {"features": [0.0], "close": 100.0}
    assert strat.on_bar(bar) is None

    stub.proba = 0.7
    sig = strat.on_bar(bar)
    assert sig and sig.side == "buy" and strat.pos_side == 1

    strat2 = MLStrategy(model=StubModel(0.3), margin=0.1)
    strat2.scaler.fit([[0.0]])
    sig = strat2.on_bar(bar)
    assert sig and sig.side == "sell" and strat2.pos_side == -1


def test_ml_strategy_risk_service_handles_stop_and_size():
    stub = StubModel(0.7)
    strat = MLStrategy(model=stub, margin=0.1)
    strat.scaler.fit([[0.0]])
    bar_open = {"features": [0.0], "close": 100.0}
    sig = strat.on_bar(bar_open)
    assert sig and sig.side == "buy"

    rm = RiskManager(risk_pct=0.02)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(rm, guard)
    svc.account.update_cash(1000.0)

    allowed, reason, delta = svc.check_order(
        "AAA", sig.side, 1000.0, bar_open["close"], strength=sig.strength
    )
    assert allowed and delta == pytest.approx(0.07)

    rm.add_fill(sig.side, delta, price=bar_open["close"])
    svc.update_position("X", "AAA", delta)

    stop_price = bar_open["close"] * (1 - rm.risk_pct) - 1
    allowed, reason, delta_stop = svc.check_order(
        "AAA", sig.side, 1000.0, stop_price
    )
    assert allowed and reason == "stop_loss"
    assert delta_stop == pytest.approx(-delta)
