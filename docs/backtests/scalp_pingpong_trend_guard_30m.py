"""30m backtest validating the updated trend guards for ScalpPingPong."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES
from tradingbot.strategies.scalp_pingpong import ScalpPingPong, ScalpPingPongConfig


SEG_A = 100
SEG_B = 40
SEG_C = 40


@dataclass
class TrendMetrics:
    trades: int = 0
    wins: int = 0
    trend_trades: int = 0
    trend_wins: int = 0
    win_rate: float = 0.0
    trend_win_rate: float = 0.0

    def accumulate(self, other: "TrendMetrics") -> None:
        self.trades += other.trades
        self.wins += other.wins
        self.trend_trades += other.trend_trades
        self.trend_wins += other.trend_wins

    def recompute(self) -> None:
        self.win_rate = self.wins / self.trades if self.trades else 0.0
        self.trend_win_rate = (
            self.trend_wins / self.trend_trades if self.trend_trades else 0.0
        )


def _synthetic_window(periods: int = 290, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(202 + seed)
    idx = pd.date_range("2021-01-01", periods=periods, freq="30min")

    seg_d = 80
    seg_e = periods - (SEG_A + SEG_B + SEG_C + seg_d)

    t_a = np.linspace(0, 6 * math.pi, SEG_A)
    returns_a = 0.0018 * np.sin(t_a) + rng.normal(0, 0.00022, SEG_A)

    returns_b = np.full(SEG_B, 0.0065) + rng.normal(0, 0.00005, SEG_B)
    returns_c = np.full(SEG_C, -0.006) + rng.normal(0, 0.00005, SEG_C)
    returns_d = rng.normal(0, 0.000004, seg_d)
    t_e = np.linspace(0, 4 * math.pi, seg_e)
    returns_e = 0.004 * np.sin(t_e) + rng.normal(0, 0.0002, seg_e)

    returns = np.concatenate([returns_a, returns_b, returns_c, returns_d, returns_e])
    prices = 19000.0 * np.exp(np.cumsum(returns))

    opens = np.concatenate(([prices[0]], prices[:-1]))
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0.0002, 0.0010, periods))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0.0002, 0.0010, periods))
    spread = rng.uniform(0.5, 1.4, periods)
    bids = prices - spread
    asks = prices + spread
    volumes = rng.uniform(55, 120, periods)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "bid": bids,
            "ask": asks,
        }
    )


def _trade_breakdown(fills: Iterable[Tuple], trend_window: set[pd.Timestamp]) -> TrendMetrics:
    metrics = TrendMetrics()
    position = 0.0
    trade_pnl = 0.0
    trade_start: pd.Timestamp | None = None
    for fill in fills:
        timestamp = fill[0]
        side = fill[2]
        qty = float(fill[4])
        realized_pnl = float(fill[10])
        if trade_start is None:
            trade_start = pd.Timestamp(timestamp)
        if side == "buy":
            position += qty
        else:
            position -= qty
        trade_pnl += realized_pnl
        if abs(position) <= 1e-9:
            metrics.trades += 1
            if trade_pnl > 0:
                metrics.wins += 1
            if trade_start in trend_window:
                metrics.trend_trades += 1
                if trade_pnl > 0:
                    metrics.trend_wins += 1
            trade_pnl = 0.0
            trade_start = None
    metrics.recompute()
    return metrics


def _run_backtest(data: Dict[str, pd.DataFrame], mode: str) -> TrendMetrics:
    original_cls = STRATEGIES[ScalpPingPong.name]

    class BaselineScalpPingPong(ScalpPingPong):
        def __init__(self, **kwargs):
            cfg = ScalpPingPongConfig(
                trend_threshold=6.0,
                trend_ma=999999,
                trend_rsi_n=600,
                trend_extreme_multiplier=1e6,
                counter_trend_strength_mult=1.0,
            )
            super().__init__(cfg=cfg, **kwargs)

    class UpdatedScalpPingPong(ScalpPingPong):
        def __init__(self, **kwargs):
            cfg = ScalpPingPongConfig(
                trend_threshold=6.0,
                trend_ma=999999,
                trend_rsi_n=600,
            )
            super().__init__(cfg=cfg, **kwargs)

    STRATEGIES[ScalpPingPong.name] = (
        BaselineScalpPingPong if mode == "baseline" else UpdatedScalpPingPong
    )
    try:
        engine = EventDrivenBacktestEngine(
            data,
            [(ScalpPingPong.name, "BTC/USDT")],
            latency=0,
            window=90,
            verbose_fills=True,
            exchange_configs={
                "default": {
                    "tick_size": 0.1,
                    "maker_fee": 0.0,
                    "taker_fee": 0.0006,
                    "market_type": "perp",
                }
            },
            timeframes={"BTC/USDT": "30m"},
            risk_pct=1.0,
        )
        result = engine.run()
    finally:
        STRATEGIES[ScalpPingPong.name] = original_cls

    trend_timestamps = set(data["BTC/USDT"]["timestamp"].iloc[SEG_A : SEG_A + SEG_B + SEG_C])
    return _trade_breakdown(result["fills"], trend_timestamps)


def main() -> None:
    seeds = [0, 1, 2]
    aggregated: Dict[str, TrendMetrics] = {
        "baseline": TrendMetrics(),
        "updated": TrendMetrics(),
    }

    for seed in seeds:
        window = _synthetic_window(seed=seed)
        data = {"BTC/USDT": window}
        aggregated["baseline"].accumulate(
            _run_backtest({k: v.copy() for k, v in data.items()}, "baseline")
        )
        aggregated["updated"].accumulate(
            _run_backtest({k: v.copy() for k, v in data.items()}, "updated")
        )

    for label, metrics in aggregated.items():
        metrics.recompute()
        print(
            f"{label}: trades={metrics.trades} wins={metrics.wins} win_rate={metrics.win_rate:.3f} "
            f"trend_trades={metrics.trend_trades} trend_win_rate={metrics.trend_win_rate:.3f}"
        )

    base = aggregated["baseline"]
    upd = aggregated["updated"]
    print("\nComparativa (trades tendencia ↓, win rate ↑ esperados):")
    print(
        f"trades Δ={upd.trades - base.trades:+d} "
        f"win_rate Δ={upd.win_rate - base.win_rate:+.3f}"
    )
    print(f"trend_trades Δ={upd.trend_trades - base.trend_trades:+d}")


if __name__ == "__main__":
    main()

