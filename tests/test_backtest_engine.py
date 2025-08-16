import numpy as np
import pandas as pd
from types import SimpleNamespace

from tradingbot.backtest.event_engine import SlippageModel, run_backtest_csv
from tradingbot.strategies import STRATEGIES


class DummyStrategy:
    name = "dummy"

    def __init__(self):
        self.i = 0

    def on_bar(self, bar):
        self.i += 1
        side = "buy" if self.i % 2 == 0 else "sell"
        return SimpleNamespace(side=side, strength=1.0)


def _make_csv(tmp_path):
    rng = pd.date_range("2021-01-01", periods=50, freq="T")
    price = np.linspace(100, 150, num=50)
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_pnl_with_and_without_slippage(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    strategies = [("dummy", "SYM")]
    data = {"SYM": str(csv_path)}

    no_slip = run_backtest_csv(data, strategies, latency=1, window=1, slippage=None)
    with_slip = run_backtest_csv(
        data, strategies, latency=1, window=1, slippage=SlippageModel(volume_impact=10.0)
    )

    assert len(no_slip["fills"]) > 0
    assert no_slip["equity"] >= with_slip["equity"]

