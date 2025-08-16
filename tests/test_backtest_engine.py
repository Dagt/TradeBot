import numpy as np
import pandas as pd
import pytest
try:  # pragma: no cover - optional dependency
    import vectorbt as vbt  # type: ignore
except Exception:  # pragma: no cover
    vbt = None
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


def test_run_vectorbt_basic():
    vbt_local = pytest.importorskip("vectorbt")
    from tradingbot.backtest.vectorbt_engine import run_vectorbt

    class MAStrategy:
        @staticmethod
        def signal(close, fast, slow):
            fast_ma = vbt_local.MA.run(close, fast)
            slow_ma = vbt_local.MA.run(close, slow)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            return entries, exits

    price = pd.Series(np.sin(np.linspace(0, 10, 100)) + 10)
    data = pd.DataFrame({"close": price})
    params = {"fast": [2, 4], "slow": [8]}

    stats = run_vectorbt(data, MAStrategy, params)
    assert not stats.empty
    assert {"sharpe_ratio", "max_drawdown", "total_return"} <= set(stats.columns)
    assert list(stats.index.names) == ["fast", "slow"]


@pytest.mark.optional
def test_vectorbt_wrapper_param_sweep():
    vbt_local = pytest.importorskip("vectorbt")
    from tradingbot.backtesting.vectorbt_wrapper import run_parameter_sweep

    def ma_signal(close, window):
        ma = vbt_local.MA.run(close, window).ma
        entries = close > ma
        exits = close < ma
        return entries, exits

    price = pd.Series(np.linspace(1, 2, 50))
    data = pd.DataFrame({"close": price})
    params = {"window": [2, 4]}
    stats = run_parameter_sweep(data, ma_signal, params)
    assert not stats.empty
    assert list(stats.index.names) == ["window"]


class OneShotStrategy:
    name = "oneshot"

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=1.0)


def test_l2_queue_partial_and_cancel(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "bid_size": [1.0, 1.0, 1.0],
            "ask_size": [0.2, 0.2, 0.4],
            "volume": 1000,
        }
    )
    path = tmp_path / "l2.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "oneshot", OneShotStrategy)
    strategies = [("oneshot", "SYM")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(),
        use_l2=True,
        cancel_unfilled=True,
    )

    assert pytest.approx(res["orders"][0]["filled"], rel=1e-9) == 0.2
    assert len(res["fills"]) == 1


def test_funding_payment(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=3, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000,
            "funding_rate": [0.0, 0.0, 0.01],
        }
    )
    path = tmp_path / "fund.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "oneshot", OneShotStrategy)
    strategies = [("oneshot", "SYM")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(data, strategies, latency=1, window=1)
    assert pytest.approx(res["funding"], rel=1e-9) == 1.0
    assert pytest.approx(res["equity"], rel=1e-9) == -1.0

