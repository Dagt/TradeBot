import pandas as pd
from types import SimpleNamespace
from tradingbot.backtesting.engine import run_backtest_csv
from tradingbot.strategies import STRATEGIES


class Signal(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class ExitStrategy:
    def __init__(self):
        self.i = 0

    def on_bar(self, bar):
        self.i += 1
        if self.i == 1:
            return Signal(side="buy", strength=1.0)
        if self.i == 2:
            return Signal(side="sell", strength=1.0)
        return Signal(side="flat")


def test_exit_signal_closes_before_liquidation(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": [100, 101, 102, 103, 104],
            "high": [100, 101, 102, 103, 104],
            "low": [100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000] * 5,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "exit_strat", ExitStrategy)
    strategies = [("exit_strat", "SYM")]
    data = {"SYM": str(path)}
    res = run_backtest_csv(data, strategies, latency=1, window=1, verbose_fills=True)

    fills = res["fills"]
    sides = [f[2] for f in fills]
    reasons = [f[1] for f in fills]
    assert "buy" in sides and "sell" in sides
    assert "liquidation" not in reasons
