import pandas as pd
import numpy as np

from tradingbot.core import Account, RiskManager as CoreRiskManager
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
        signal = strat.on_bar({"window": prices.iloc[: i + 1], "volatility": 0})
    assert meta_model.fit_called
    assert signal is None

    # allow trades by setting meta model to return 1
    meta_model2 = DummyModel(value=1)
    primary_model2 = DummyModel(value=1)
    strat2 = TripleBarrier(
        horizon=2, upper_pct=0.02, lower_pct=0.02, training_window=5, meta_model=meta_model2
    )
    strat2.model = primary_model2
    signal = None
    for i in range(len(prices)):
        signal = strat2.on_bar({"window": prices.iloc[: i + 1], "volatility": 0})
        if signal is not None:
            break
    assert signal is not None
    assert signal.side == "buy"


def test_triple_barrier_scalping_exit():
    prices = pd.DataFrame({"close": [100, 104, 97, 103]})
    primary_model = DummyModel(value=1)
    meta_model = DummyModel(value=1)
    strat = TripleBarrier(
        horizon=1,
        upper_pct=0.02,
        lower_pct=0.02,
        training_window=3,
        meta_model=meta_model,
    )
    strat.model = primary_model
    signals = []
    for i in range(len(prices)):
        signals.append(strat.on_bar({"window": prices.iloc[: i + 1], "volatility": 0}))
    assert signals[2] is not None and signals[2].side == "buy"
    trade = {"side": "buy", "entry_price": 97.0, "qty": 1.0, "stop": 96.0}
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    rm.update_trailing(trade, 96.0)
    decision = rm.manage_position({**trade, "current_price": 96.0})
    assert decision == "close"


def test_triple_barrier_loads_config(tmp_path):
    cfg = tmp_path / "tb.yaml"
    cfg.write_text(
        """
horizon: 3
upper_pct: 0.05
lower_pct: 0.01
training_window: 50
"""
    )
    strat = TripleBarrier(config_path=str(cfg))
    assert strat.horizon == 3
    assert strat.upper_pct == 0.05
    assert strat.lower_pct == 0.01
    assert strat.training_window == 50

