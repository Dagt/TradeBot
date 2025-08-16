import numpy as np
import pandas as pd
import pytest

vbt_local = pytest.importorskip("vectorbt")

from tradingbot.backtest.vectorbt_engine import walk_forward, optimize_vectorbt_optuna


class MAStrategy:
    @staticmethod
    def signal(close, fast, slow):
        fast_ma = vbt_local.MA.run(close, fast)
        slow_ma = vbt_local.MA.run(close, slow)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits


def _make_data():
    price = pd.Series(np.sin(np.linspace(0, 6, 60)) + 10)
    return pd.DataFrame({"close": price})


def test_walk_forward_basic():
    data = _make_data()
    params = {"fast": [2, 4], "slow": [8]}
    res = walk_forward(data, MAStrategy, params, train_size=20, test_size=10)
    assert not res.empty
    assert {"sharpe_ratio", "max_drawdown", "total_return", "dsr"} <= set(res.columns)
    assert res["dsr"].between(0, 1).all()


def test_optimize_vectorbt_optuna():
    data = _make_data()
    space = {"fast": {"type": "int", "low": 2, "high": 4}, "slow": {"type": "int", "low": 8, "high": 8}}
    study = optimize_vectorbt_optuna(data, MAStrategy, space, n_trials=2)
    assert len(study.trials) == 2
