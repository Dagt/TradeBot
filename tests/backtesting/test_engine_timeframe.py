import pandas as pd
import pytest

from tradingbot.backtesting import engine as engine_module


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


def test_event_engine_includes_timeframe_in_df_window(monkeypatch):
    timeframe = "15m"
    captured: list[dict] = []

    class RecordingStrategy:
        def __init__(self, risk_service=None):  # noqa: ANN001
            self.risk_service = risk_service

        def on_bar(self, bar):  # noqa: ANN001
            captured.append(bar)
            raise RuntimeError("stop")

    monkeypatch.setitem(engine_module.STRATEGIES, "recording_tf", RecordingStrategy)
    data = {"SYM": _sample_frame()}
    eng = engine_module.EventDrivenBacktestEngine(
        data,
        [("recording_tf", "SYM")],
        window=1,
        timeframes={"SYM": timeframe},
    )

    with pytest.raises(RuntimeError):
        eng.run()

    assert captured, "strategy should receive at least one bar"
    assert captured[0]["timeframe"] == timeframe
    assert captured[0]["symbol"] == "SYM"
    assert "window" in captured[0]


def test_event_engine_includes_timeframe_in_array_window(monkeypatch):
    timeframe = "15m"
    captured: list[dict] = []

    class ArrayStrategy:
        needs_window_df = False

        def __init__(self, risk_service=None):  # noqa: ANN001
            self.risk_service = risk_service

        def on_bar(self, bar):  # noqa: ANN001
            captured.append(bar)
            raise RuntimeError("stop")

    monkeypatch.setitem(engine_module.STRATEGIES, "array_tf", ArrayStrategy)
    data = {"SYM": _sample_frame()}
    eng = engine_module.EventDrivenBacktestEngine(
        data,
        [("array_tf", "SYM")],
        window=1,
        timeframes={"SYM": timeframe},
    )

    with pytest.raises(RuntimeError):
        eng.run()

    assert captured, "strategy should receive at least one bar"
    bar = captured[0]
    assert bar["timeframe"] == timeframe
    assert bar["symbol"] == "SYM"
    assert "symbol" in bar
