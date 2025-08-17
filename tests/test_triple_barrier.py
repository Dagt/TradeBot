import pandas as pd
import numpy as np

from tradingbot.strategies.triple_barrier import (
    TripleBarrier,
    apply_meta_labeling,
)


class DummyModel:
    """Simple model returning a constant prediction."""

    def __init__(self, value: int = 1):
        self.value = value
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def test_apply_meta_labeling_generates_binary_labels():
    labels = pd.Series([1, 0, -1, 1], index=pd.RangeIndex(4))
    features = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4]})
    model = DummyModel(value=1)
    preds = apply_meta_labeling(labels, features, model)
    assert model.fit_called
    assert set(preds.unique()) == {1}
    assert len(preds) == len(labels)


def test_triple_barrier_meta_filtering():
    prices = pd.DataFrame({"close": [100, 103, 97, 104, 96, 100]})

    primary_model = DummyModel(value=1)
    meta_model = DummyModel(value=0)
    strat = TripleBarrier(
        horizon=2, upper_pct=0.02, lower_pct=0.02, training_window=5, meta_model=meta_model
    )
    strat.model = primary_model

    signal = None
    for i in range(len(prices)):
        signal = strat.on_bar({"window": prices.iloc[: i + 1]})
    assert meta_model.fit_called
    assert signal is not None
    assert signal.side == "flat"

    # allow trades by setting meta model to return 1
    meta_model2 = DummyModel(value=1)
    primary_model2 = DummyModel(value=1)
    strat2 = TripleBarrier(
        horizon=2, upper_pct=0.02, lower_pct=0.02, training_window=5, meta_model=meta_model2
    )
    strat2.model = primary_model2
    signal = None
    for i in range(len(prices)):
        signal = strat2.on_bar({"window": prices.iloc[: i + 1]})
    assert signal is not None
    assert signal.side == "buy"
