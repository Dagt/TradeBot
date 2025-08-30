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
    assert sig and sig.side == "buy" and strat.trade["side"] == "buy"

    strat2 = MLStrategy(model=StubModel(0.3), margin=0.1)
    strat2.scaler.fit([[0.0]])
    sig = strat2.on_bar(bar)
    assert sig and sig.side == "sell" and strat2.trade["side"] == "sell"
