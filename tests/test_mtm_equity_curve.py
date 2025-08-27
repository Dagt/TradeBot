import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import run_backtest_csv
from tradingbot.strategies import STRATEGIES


class BuyHoldStrategy:
    name = "buyhold"

    def __init__(self):
        self.done = False

    def on_bar(self, bar):
        if not self.done:
            self.done = True
            return SimpleNamespace(side="buy", strength=1.0)
        return None


def test_equity_curve_marks_open_positions(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=4, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": [100, 100, 110, 120],
            "high": [100, 100, 110, 120],
            "low": [100, 100, 110, 120],
            "close": [100, 100, 110, 120],
            "volume": [1000, 1000, 1000, 1000],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "buyhold", BuyHoldStrategy)
    strategies = [("buyhold", "SYM")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(
        data, strategies, latency=0, window=0, initial_equity=0
    )
    assert res["equity_curve"] == pytest.approx([0.0, 0.0, 0.0, 10.0, 20.0, 20.0])
