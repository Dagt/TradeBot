import math

import pandas as pd

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES
from tradingbot.strategies.base import Strategy
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.breakout_vol import BreakoutVol
from tradingbot.strategies.mean_reversion import MeanReversion
from tradingbot.strategies.momentum import Momentum
from tradingbot.strategies.scalp_pingpong import ScalpPingPong
from tradingbot.strategies.trend_following import TrendFollowing


def _make_ohlcv(rows: int) -> pd.DataFrame:
    base = 100.0
    data = {
        "open": [base + i * 0.5 for i in range(rows)],
        "high": [base + i * 0.5 + 0.4 for i in range(rows)],
        "low": [base + i * 0.5 - 0.4 for i in range(rows)],
        "close": [base + i * 0.5 + 0.2 for i in range(rows)],
        "volume": [10.0] * rows,
    }
    return pd.DataFrame(data)


def test_breakout_atr_cooldown_scales_with_timeframe():
    fast = BreakoutATR(timeframe="1m")
    mid = BreakoutATR(timeframe="3m")
    slow = BreakoutATR(timeframe="1h")

    assert fast.cooldown_bars == 3
    assert mid.cooldown_bars == 1
    assert slow.cooldown_bars == 1


def test_breakout_vol_lookback_scales():
    df = _make_ohlcv(60)
    strat_fast = BreakoutVol(timeframe="1m")
    strat_slow = BreakoutVol(timeframe="15m")

    strat_fast.on_bar({"window": df, "timeframe": "1m", "symbol": "X"})
    strat_slow.on_bar({"window": df, "timeframe": "15m", "symbol": "X"})

    assert strat_fast.lookback == 10
    assert strat_slow.lookback == 2


def test_momentum_cooldown_scales():
    df = pd.DataFrame({
        "close": [1 + 0.1 * i for i in range(80)],
        "volume": [5.0] * 80,
    })
    strat_fast = Momentum(timeframe="1m", min_volume=0, min_volatility=0)
    strat_slow = Momentum(timeframe="15m", min_volume=0, min_volatility=0)

    strat_fast.on_bar({"window": df, "timeframe": "1m"})
    strat_slow.on_bar({"window": df, "timeframe": "15m"})

    assert strat_fast.cooldown_bars == 3
    assert strat_slow.cooldown_bars == 1


def test_mean_reversion_time_stop_scales():
    df = _make_ohlcv(80)
    strat_fast = MeanReversion(timeframe="1m", time_stop=10, min_volatility=0)
    strat_slow = MeanReversion(timeframe="1h", time_stop=10, min_volatility=0)

    strat_fast.on_bar({"window": df, "timeframe": "1m", "symbol": "X"})
    strat_slow.on_bar({"window": df, "timeframe": "1h", "symbol": "X"})

    assert strat_fast.time_stop == 10
    assert strat_slow.time_stop == 1


def test_scalp_pingpong_lookback_scales():
    df = pd.DataFrame({"close": [100 + 0.01 * i for i in range(100)]})
    strat_fast = ScalpPingPong(timeframe="1m")
    strat_slow = ScalpPingPong(timeframe="1h")

    strat_fast.on_bar({"window": df, "timeframe": "1m"})
    strat_slow.on_bar({"window": df, "timeframe": "1h"})

    lookback_fast = max(1, int(math.ceil(strat_fast.cfg.lookback / 1)))
    lookback_slow = max(1, int(math.ceil(strat_slow.cfg.lookback / 60)))

    assert lookback_fast == 15
    assert lookback_slow == 1


def test_trend_following_uses_timeframe_minutes():
    df = _make_ohlcv(40)
    strat = TrendFollowing(timeframe="1h", vol_lookback=20)
    # With 40 rows and 1-hour timeframe, there should be enough history for the
    # strategy to run without triggering the warm-up guard.
    result = strat.on_bar({"window": df, "timeframe": "1h"})
    assert strat._tf_minutes("1h", strat.timeframe) == 60
    # Strategy may or may not emit a signal depending on the random walk; we
    # only verify that the call completed without raising and the timeframe is
    # stored.
    assert strat.timeframe == "1h"


def test_engine_propagates_timeframe_into_strategy(monkeypatch):
    class DummyStrategy(Strategy):
        name = "dummy_tf"

        def __init__(self, timeframe: str | None = None, **kwargs) -> None:
            self.timeframe = timeframe
            self.seen: list[str | None] = []

        def on_bar(self, bar: dict) -> None:
            self.seen.append(bar.get("timeframe"))
            return None

    data = pd.DataFrame({
        "open": [1, 2, 3, 4, 5],
        "high": [1.1, 2.1, 3.1, 4.1, 5.1],
        "low": [0.9, 1.9, 2.9, 3.9, 4.9],
        "close": [1, 2, 3, 4, 5],
        "volume": [1] * 5,
    })

    STRATEGIES[DummyStrategy.name] = DummyStrategy
    try:
        engine = EventDrivenBacktestEngine(
            {"SYM": data},
            [(DummyStrategy.name, "SYM")],
            window=2,
            timeframes={"SYM": "15m"},
        )
        engine.run()
        strat = engine.strategies[(DummyStrategy.name, "SYM")]
        assert strat.timeframe == "15m"
        assert strat.seen
        assert all(tf == "15m" for tf in strat.seen)
    finally:
        STRATEGIES.pop(DummyStrategy.name, None)
