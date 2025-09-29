from __future__ import annotations

import pandas as pd

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies.base import Strategy, Signal
from tradingbot.strategies import STRATEGIES


def test_min_strength_fraction_filters_small_signals(monkeypatch):
    class StrengthSequenceBase(Strategy):
        name = "seq_base"
        max_signal_strength = 1.0
        strengths: list[float] = []
        min_strength_fraction = 0.05

        def __init__(self, *, risk_service=None, timeframe: str | None = None, **kwargs):
            super().__init__(**kwargs)
            self.risk_service = None
            if timeframe is not None:
                self.timeframe = timeframe
            self._i = 0

        def _next_raw(self) -> float | None:
            if self._i >= len(self.strengths):
                return None
            raw = self.strengths[self._i]
            self._i += 1
            return raw

        def _normalise(self, raw: float) -> float | None:
            raise NotImplementedError

        def on_bar(self, bar: dict) -> Signal | None:
            raw = self._next_raw()
            if raw is None or raw <= 0:
                return None
            price = float(bar["window"]["close"].iloc[-1])
            strength = self._normalise(raw)
            if strength is None:
                return None
            sig = Signal("buy", strength)
            sig.limit_price = price
            return sig

    class SequentialHFT(StrengthSequenceBase):
        name = "sequential_hft"

        def _normalise(self, raw: float) -> float | None:
            if self.max_signal_strength <= 0:
                return None
            strength = raw / self.max_signal_strength
            if strength < self.min_strength_fraction:
                return None
            return min(1.0, strength)

    class LegacySequentialHFT(StrengthSequenceBase):
        name = "legacy_sequential_hft"

        def _normalise(self, raw: float) -> float | None:
            if self.max_signal_strength <= 0:
                return None
            strength = raw / self.max_signal_strength
            strength = max(self.min_strength_fraction, strength)
            return min(1.0, strength)

    monkeypatch.setitem(STRATEGIES, SequentialHFT.name, SequentialHFT)
    monkeypatch.setitem(STRATEGIES, LegacySequentialHFT.name, LegacySequentialHFT)

    strengths = [0.02, 0.06, 0.5, 0.04, 0.2]
    SequentialHFT.strengths = strengths.copy()
    LegacySequentialHFT.strengths = strengths.copy()

    length = len(strengths) + 2
    closes = [100 + i * 0.1 for i in range(length)]
    data = pd.DataFrame(
        {
            "timestamp": range(length),
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
            "volume": [1_000] * length,
        }
    )

    engine_new = EventDrivenBacktestEngine(
        {"SYM": data},
        [(SequentialHFT.name, "SYM")],
        window=1,
        latency=0,
        verbose_fills=True,
        risk_pct=0.01,
    )
    result_new = engine_new.run()

    engine_legacy = EventDrivenBacktestEngine(
        {"SYM": data},
        [(LegacySequentialHFT.name, "SYM")],
        window=1,
        latency=0,
        verbose_fills=True,
        risk_pct=0.01,
    )
    result_legacy = engine_legacy.run()

    fills_new = [fill for fill in result_new["fills"] if fill[1] == "order"]
    fills_legacy = [fill for fill in result_legacy["fills"] if fill[1] == "order"]

    below_threshold = sum(1 for raw in strengths if raw < SequentialHFT.min_strength_fraction)
    assert len(fills_new) == len(strengths) - below_threshold
    assert len(fills_new) < len(fills_legacy)
