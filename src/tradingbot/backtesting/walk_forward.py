from __future__ import annotations

"""Walk-forward backtesting utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import pandas as pd

from .engine import EventDrivenBacktestEngine
from ..strategies import STRATEGIES
from ..reporting.metrics import evaluate


@dataclass
class SplitResult:
    """Summary of a walk-forward split."""

    split: int
    params: Dict[str, Any]
    metrics: Dict[str, float]


def walk_forward_splits(n: int, train: int, test: int) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield sequential train/test index splits."""

    start = 0
    while start + train + test <= n:
        train_idx = list(range(start, start + train))
        test_idx = list(range(start + train, start + train + test))
        yield train_idx, test_idx
        start += test


def walk_forward_backtest(
    csv_path: str,
    symbol: str,
    strategy_name: str,
    param_grid: Sequence[Dict[str, Any]],
    train_size: int,
    test_size: int,
    latency: int = 1,
    window: int = 120,
    verbose_fills: bool = False,
    fills_csv: str | None = None,
    exchange_configs: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Run a basic walk-forward analysis and return metrics for each split."""

    df = pd.read_csv(csv_path)
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")

    if fills_csv:
        verbose_fills = False

    records: List[Dict[str, Any]] = []
    split_no = 0
    for train_idx, test_idx in walk_forward_splits(len(df), train_size, test_size):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        best_params: Dict[str, Any] | None = None
        best_equity = -float("inf")
        for params in param_grid:
            strat = strat_cls(**params)
            engine = EventDrivenBacktestEngine(
                {symbol: train_df},
                [(strategy_name, symbol)],
                latency=latency,
                window=window,
                verbose_fills=verbose_fills,
                exchange_configs=exchange_configs,
            )
            engine.strategies[(strategy_name, symbol)] = strat
            res = engine.run()
            eq = float(res.get("equity", 0.0))
            if eq > best_equity:
                best_equity = eq
                best_params = params

        strat = strat_cls(**(best_params or {}))
        engine = EventDrivenBacktestEngine(
            {symbol: test_df},
            [(strategy_name, symbol)],
            latency=latency,
            window=window,
            verbose_fills=verbose_fills,
            exchange_configs=exchange_configs,
        )
        engine.strategies[(strategy_name, symbol)] = strat
        test_res = engine.run(fills_csv=fills_csv)
        metrics = evaluate(test_res.get("equity_curve", []))

        rec = {"split": split_no, "params": best_params}
        rec.update(metrics)
        records.append(rec)
        split_no += 1

    return pd.DataFrame(records)


__all__ = [
    "walk_forward_backtest",
    "walk_forward_splits",
]
