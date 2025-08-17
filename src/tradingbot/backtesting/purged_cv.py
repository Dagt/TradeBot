"""Purged cross-validation utilities.

This module provides a simple generator to create purged cross-validation
splits as described by López de Prado.  It also exposes a helper to run a
cross‑validation cycle on a strategy class with arbitrary parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence, Tuple, Type, Any, Dict

import pandas as pd


__all__ = ["purged_splits", "run_purged_cv"]


def purged_splits(
    n_samples: int,
    n_splits: int,
    embargo: float = 0.0,
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Yield purged cross‑validation splits.

    Parameters
    ----------
    n_samples:
        Total number of observations.
    n_splits:
        Number of folds.
    embargo:
        Fraction of observations to exclude on each side of the test set to
        avoid look‑ahead bias.  ``embargo=0.01`` removes 1% of samples before
        and after the test window from the training indices.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    indices = list(range(n_samples))
    fold_sizes = [n_samples // n_splits] * n_splits
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1

    embargo_size = int(n_samples * embargo)
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx: List[int] = []
        left_end = max(0, start - embargo_size)
        if left_end > 0:
            train_idx.extend(indices[:left_end])
        right_start = min(n_samples, stop + embargo_size)
        if right_start < n_samples:
            train_idx.extend(indices[right_start:])
        yield train_idx, test_idx
        current = stop


@dataclass
class _FoldResult:
    """Internal container to store per‑fold metrics."""

    train_idx: Sequence[int]
    test_idx: Sequence[int]
    equity: float


def run_purged_cv(
    data: pd.DataFrame,
    strategy_cls: Type[Any],
    params: Dict[str, Any] | None = None,
    *,
    n_splits: int = 5,
    embargo: float = 0.0,
) -> Dict[str, Any]:
    """Execute purged cross‑validation for ``strategy_cls``.

    The ``strategy_cls`` is expected to implement ``fit`` and ``predict``
    methods.  ``fit`` receives the training ``DataFrame`` while ``predict``
    receives the test ``DataFrame`` and must return an iterable of position
    values (e.g. ``1`` for long, ``-1`` for short, ``0`` for flat).  The
    evaluation computes the cumulative return assuming positions are held for
    one period.

    Returns
    -------
    dict
        Mapping containing the list of fold results under ``folds`` and the
        average test equity as ``purged_cv``.
    """

    params = params or {}
    fold_results: List[_FoldResult] = []

    for train_idx, test_idx in purged_splits(len(data), n_splits, embargo):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        strategy = strategy_cls(**params)
        fit = getattr(strategy, "fit", None)
        if callable(fit):
            fit(train_df)

        predict = getattr(strategy, "predict", None)
        if not callable(predict):
            raise AttributeError("strategy_cls must define a 'predict' method")
        preds = pd.Series(predict(test_df), index=test_df.index).astype(float)

        # Align predictions with next period returns
        returns = test_df["close"].pct_change().shift(-1).fillna(0.0)
        equity = float((1.0 + preds * returns).prod())
        fold_results.append(_FoldResult(train_idx, test_idx, equity))

    avg_equity = (
        sum(fr.equity for fr in fold_results) / len(fold_results)
        if fold_results
        else 0.0
    )

    return {
        "folds": [fr.__dict__ for fr in fold_results],
        "purged_cv": avg_equity,
    }
