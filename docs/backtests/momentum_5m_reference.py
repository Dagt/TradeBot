#!/usr/bin/env python3
"""Reference 5m backtest comparing momentum limit logic revisions."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

# Ensure project sources are importable when executing from this file.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES
import tradingbot.strategies.momentum as momentum_module
from tradingbot.strategies.momentum import Momentum


@dataclass
class BacktestResult:
    fills: int
    avg_fee: float
    orders: int


def _synthetic_5m_window(periods: int = 240, seed: int = 7) -> pd.DataFrame:
    """Return a reproducible synthetic 5m window."""

    rng = np.random.RandomState(seed)
    base_price = 20_000.0
    steps = rng.normal(scale=25.0, size=periods)
    closes = base_price + np.cumsum(steps)
    opens = np.concatenate(([base_price], closes[:-1]))
    spread = rng.uniform(0.5, 1.5, size=periods)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(scale=7.5, size=periods))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(scale=7.5, size=periods))
    volumes = rng.uniform(5.0, 15.0, size=periods) * 100.0
    index = pd.date_range("2021-01-01", periods=periods, freq="5min")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "bid": closes - spread,
            "ask": closes + spread,
        }
    )
    return frame


def _legacy_config(side: str, price: float, anchor: float, atr_val: float, bar: Dict[str, float]):
    limit_span = max(price * 0.001, atr_val * 0.5)
    limit_span = max(limit_span, abs(price - anchor))
    limit_span = max(limit_span, price * 0.0005)
    if not math.isfinite(limit_span) or limit_span <= 0:
        limit_span = max(abs(price) * 0.0005, 1e-6)

    if side == "buy":
        base_price = max(0.0, anchor - limit_span)
    else:
        base_price = anchor + limit_span

    initial_offset = max(limit_span * 0.4, price * 0.0003, atr_val * 0.25)
    initial_offset = min(initial_offset, limit_span)
    step_offset = max(limit_span * 0.25, price * 0.0002)
    step_offset = min(step_offset, limit_span)
    maker_initial = max(price * 0.0002, min(initial_offset * 0.5, limit_span))

    direction = 1.0 if side == "buy" else -1.0
    limit_price = base_price + direction * initial_offset
    if side == "buy":
        limit_price = min(limit_price, anchor)
    else:
        limit_price = max(limit_price, anchor)

    meta = {
        "base_price": base_price,
        "limit_offset": abs(limit_span),
        "initial_offset": abs(initial_offset),
        "offset_step": abs(step_offset),
        "max_offset": abs(limit_span),
        "maker_initial_offset": abs(maker_initial),
        "maker_patience": 1,
        "step_mult": 0.5,
        "chase": True,
        "post_only": True,
        "anchor_price": anchor,
    }
    return limit_price, meta


class BacktestMomentum(Momentum):
    """Momentum variant with deterministic defaults for the comparison."""

    name = "momentum"

    def __init__(self, **kwargs):
        kwargs.setdefault("fast_ema", 12)
        kwargs.setdefault("slow_ema", 26)
        kwargs.setdefault("rsi_n", 14)
        kwargs.setdefault("atr_n", 14)
        kwargs.setdefault("roc_n", 4)
        kwargs.setdefault("vol_window", 30)
        kwargs.setdefault("min_volume", 0)
        kwargs.setdefault("min_volatility", 0)
        kwargs.setdefault("use_rsi", True)
        super().__init__(**kwargs)
        self.risk_service = None


def _run(helper: Callable) -> BacktestResult:
    original = momentum_module._configure_limit
    momentum_module._configure_limit = helper
    original_cls = STRATEGIES["momentum"]
    STRATEGIES["momentum"] = BacktestMomentum
    try:
        engine = EventDrivenBacktestEngine(
            {"BTC/USDT": _synthetic_5m_window()},
            [("momentum", "BTC/USDT")],
            latency=0,
            window=120,
            verbose_fills=True,
            exchange_configs={"default": {"tick_size": 0.1, "maker_fee": 0.0002, "taker_fee": 0.0006}},
            timeframes={"BTC/USDT": "5m"},
        )
        result = engine.run()
        fills = int(result["fill_count"])
        fee_total = sum(fill[8] for fill in result["fills"])
        avg_fee = fee_total / fills if fills else 0.0
        return BacktestResult(fills=fills, avg_fee=avg_fee, orders=int(result["order_count"]))
    finally:
        STRATEGIES["momentum"] = original_cls
        momentum_module._configure_limit = original


def main() -> None:
    updated = _run(momentum_module._configure_limit)
    legacy = _run(_legacy_config)
    print("Legacy:", legacy)
    print("Updated:", updated)


if __name__ == "__main__":
    main()
