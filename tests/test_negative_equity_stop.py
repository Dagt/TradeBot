import logging
import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import run_backtest_csv
from tradingbot.strategies import STRATEGIES


class AlwaysBuyStrategy:
    name = "always_buy"

    def on_bar(self, bar):
        return SimpleNamespace(side="buy", strength=1.0)


def test_engine_stops_on_negative_equity_and_clamps_drawdown(tmp_path, monkeypatch, caplog):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": [1, 1, 1, -1, 1],
            "high": [1, 1, 1, -1, 1],
            "low": [1, 1, 1, -1, 1],
            "close": [1, 1, 1, -1, 1],
            "volume": [1000] * 5,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "always_buy", AlwaysBuyStrategy)
    strategies = [("always_buy", "SYM")]
    data = {"SYM": str(path)}

    caplog.set_level(logging.WARNING)
    res = run_backtest_csv(data, strategies, latency=1, window=1, initial_equity=100.0)

    assert len(res["equity_curve"]) < len(df) + 1
    assert any("Equity depleted" in m for m in caplog.messages)
    assert res["max_drawdown"] <= 1.0
