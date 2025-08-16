import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


class DummyStrategy:
    name = "dummy"

    def on_bar(self, bar):
        return SimpleNamespace(side="buy", strength=1.0)


@pytest.mark.integration
def test_event_engine_runs(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    price = np.linspace(100, 104, num=5)
    df = pd.DataFrame({
        "timestamp": rng.view("int64") // 10**9,
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price,
        "volume": 1000,
    })
    data = {"SYM": df}
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    engine = EventDrivenBacktestEngine(data, [("dummy", "SYM")], latency=1, window=1)
    res = engine.run()
    assert "equity" in res
    assert len(res["fills"]) > 0
