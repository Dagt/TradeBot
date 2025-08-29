import numpy as np

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


def test_ml_strategy_exit_tp_sl_rev():
    stub = StubModel(0.7)
    strat = MLStrategy(model=stub, margin=0.1, tp_pct=0.02, sl_pct=0.02)
    strat.scaler.fit([[0.0]])
    bar_open = {"features": [0.0], "close": 100.0}
    strat.on_bar(bar_open)

    bar_tp = {"features": [0.0], "close": 102.0}
    sig = strat.on_bar(bar_tp)
    assert sig and sig.side == "sell" and strat.pos_side == 0

    strat.on_bar(bar_open)
    bar_sl = {"features": [0.0], "close": 98.0}
    sig = strat.on_bar(bar_sl)
    assert sig and sig.side == "sell" and strat.pos_side == 0

    strat.on_bar(bar_open)
    stub.proba = 0.2
    bar_rev = {"features": [0.0], "close": 100.0}
    sig = strat.on_bar(bar_rev)
    assert sig and sig.side == "sell" and strat.pos_side == 0

    strat_short = MLStrategy(model=StubModel(0.3), margin=0.1, tp_pct=0.02, sl_pct=0.02)
    strat_short.scaler.fit([[0.0]])
    strat_short.on_bar(bar_open)
    stub2 = strat_short.model
    assert isinstance(stub2, StubModel)
    stub2.proba = 0.8
    sig = strat_short.on_bar(bar_rev)
    assert sig and sig.side == "buy" and strat_short.pos_side == 0
