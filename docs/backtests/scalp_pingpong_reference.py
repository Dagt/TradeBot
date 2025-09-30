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


@dataclass
class BacktestMetrics:
    fills: int = 0
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    fee_total: float = 0.0
    pnl: float = 0.0

    def accumulate(self, other: "BacktestMetrics") -> None:
        self.fills += other.fills
        self.trades += other.trades
        self.wins += other.wins
        self.fee_total += other.fee_total
        self.pnl += other.pnl

    def recompute(self) -> None:
        self.win_rate = self.wins / self.trades if self.trades else 0.0


def _synthetic_window(tf_minutes: int, periods: int = 290, seed: int = 0) -> pd.DataFrame:
    """Create a deterministic OHLCV window with trend and low-volatility regimes."""

    rng = np.random.RandomState(10 + tf_minutes + seed * 7)
    idx = pd.date_range("2021-01-01", periods=periods, freq=f"{tf_minutes}min")

    seg_a = 100
    seg_b = 40
    seg_c = 40
    seg_d = 80
    seg_e = periods - (seg_a + seg_b + seg_c + seg_d)

    t_a = np.linspace(0, 6 * math.pi, seg_a)
    returns_a = 0.0022 * np.sin(t_a) + rng.normal(0, 0.00025, seg_a)

    returns_b = np.full(seg_b, 0.0013) + rng.normal(0, 0.00018, seg_b)
    returns_c = np.full(seg_c, -0.0011) + rng.normal(0, 0.0002, seg_c)
    returns_d = rng.normal(0, 0.000004, seg_d)
    t_e = np.linspace(0, 4 * math.pi, seg_e)
    returns_e = 0.0045 * np.sin(t_e) + rng.normal(0, 0.0002, seg_e)

    returns = np.concatenate([returns_a, returns_b, returns_c, returns_d, returns_e])
    prices = 18000.0 * np.exp(np.cumsum(returns))

    opens = np.concatenate(([prices[0]], prices[:-1]))
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0.0002, 0.0010, periods))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0.0002, 0.0010, periods))
    spread = rng.uniform(0.5, 1.5, periods)
    bids = prices - spread
    asks = prices + spread
    volumes = rng.uniform(50, 120, periods) * (1 + tf_minutes / 100.0)

    frame = pd.DataFrame(
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
    return frame


def _compute_trade_stats(fills: Iterable[Tuple]) -> Tuple[int, int, float]:
    position = 0.0
    realized_acc = 0.0
    trades: list[float] = []
    fee_total = 0.0
    for fill in fills:
        # Tuple layout documented in EventDrivenBacktestEngine.run
        _, _, side, _, qty, _, _, _, fee_cost, _, realized_pnl, _, _ = fill
        qty = float(qty)
        realized_pnl = float(realized_pnl)
        fee_total += float(fee_cost)
        if side == "buy":
            position += qty
        else:
            position -= qty
        realized_acc += realized_pnl
        if abs(position) <= 1e-9 and abs(realized_acc) > 0:
            trades.append(realized_acc)
            realized_acc = 0.0
    trade_count = len(trades)
    wins = sum(1 for pnl in trades if pnl > 0)
    return trade_count, wins, fee_total


def _run_backtest(
    data: Dict[str, pd.DataFrame],
    timeframe: str,
    mode: str,
) -> BacktestMetrics:
    original_cls = STRATEGIES[ScalpPingPong.name]

    class BaselineScalpPingPong(ScalpPingPong):
        def __init__(self, **kwargs):
            cfg = ScalpPingPongConfig(
                min_volatility=0.0,
                min_volatility_quantile=0.0,
                min_volatility_window_mult=1.0,
                min_volatility_fallbacks={1.0: 0.0, 5.0: 0.0, 15.0: 0.0, 30.0: 0.0, 60.0: 0.0},
                trend_penalty_pct=0.0,
            )
            super().__init__(cfg=cfg, **kwargs)

    class UpdatedScalpPingPong(ScalpPingPong):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

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
            timeframes={"BTC/USDT": timeframe},
            risk_pct=1.0,
        )
        result = engine.run()
    finally:
        STRATEGIES[ScalpPingPong.name] = original_cls

    trade_count, win_count, fee_total = _compute_trade_stats(result["fills"])
    pnl = float(result.get("pnl", 0.0))
    return BacktestMetrics(
        fills=int(result.get("fill_count", 0)),
        trades=trade_count,
        wins=win_count,
        win_rate=(win_count / trade_count if trade_count else 0.0),
        fee_total=fee_total,
        pnl=pnl,
    )


def main() -> None:
    timeframes = ["5m", "15m", "30m"]
    seeds = [0, 1, 2]
    aggregated: Dict[str, Dict[str, BacktestMetrics]] = {}

    for tf in timeframes:
        aggregated[tf] = {"baseline": BacktestMetrics(), "updated": BacktestMetrics()}
        minutes = int(tf.rstrip("m"))
        for seed in seeds:
            window = _synthetic_window(minutes, seed=seed)
            data = {"BTC/USDT": window}
            base = _run_backtest({k: v.copy() for k, v in data.items()}, tf, "baseline")
            upd = _run_backtest({k: v.copy() for k, v in data.items()}, tf, "updated")
            aggregated[tf]["baseline"].accumulate(base)
            aggregated[tf]["updated"].accumulate(upd)

    for tf in timeframes:
        print(f"=== {tf} timeframe ===")
        for label in ("baseline", "updated"):
            metrics = aggregated[tf][label]
            metrics.recompute()
            print(
                f"{label}: trades={metrics.trades} fills={metrics.fills} "
                f"wins={metrics.wins} win_rate={metrics.win_rate:.3f} "
                f"fees={metrics.fee_total:.4f} pnl={metrics.pnl:.2f}"
            )

    print("\nSummary (win_rate↑, fees↓ across seeds):")
    for tf in timeframes:
        base = aggregated[tf]["baseline"]
        upd = aggregated[tf]["updated"]
        delta_win = upd.win_rate - base.win_rate
        delta_fee = upd.fee_total - base.fee_total
        print(
            f"{tf}: win_rate Δ={delta_win:+.3f} fee Δ={delta_fee:+.4f} (base={base.win_rate:.3f}/{base.fee_total:.4f}, "
            f"updated={upd.win_rate:.3f}/{upd.fee_total:.4f})"
        )


if __name__ == "__main__":
    main()
