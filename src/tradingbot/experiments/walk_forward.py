from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

import mlflow

from .mlflow_utils import start_run
from ..backtesting.engine import walk_forward_optimize


def run_walk_forward_experiment(
    csv_path: str,
    symbol: str,
    strategy: str,
    param_grid: List[Dict[str, Any]],
    *,
    train_size: int | None = None,
    test_size: int | None = None,
    n_splits: int | None = None,
    embargo: float = 0.0,
    experiment: str = "walk-forward",
    latency: int = 1,
    window: int = 120,
) -> Dict[str, Any]:
    """Run walk-forward optimisation and log results to MLflow.

    The function executes :func:`walk_forward_optimize` using the provided
    parameters and records the metrics of each split in individual MLflow runs.
    An aggregate run summarises the average test metrics to ease comparison in
    the MLflow UI.
    """

    results = walk_forward_optimize(
        csv_path,
        symbol,
        strategy,
        param_grid,
        train_size=train_size,
        test_size=test_size,
        latency=latency,
        window=window,
        n_splits=n_splits,
        embargo=embargo,
    )

    if not results:
        return {"results": [], "avg_test_equity": 0.0}

    with start_run(experiment, run_name="walk_forward"):
        for rec in results:
            with start_run(
                experiment,
                run_name=f"split_{rec['split']}",
                params=rec.get("params"),
                nested=True,
            ):
                mlflow.log_metric("train_equity", float(rec.get("train_equity", 0.0)))
                mlflow.log_metric("test_equity", float(rec.get("test_equity", 0.0)))
                if "train_sharpe" in rec:
                    mlflow.log_metric("train_sharpe", float(rec["train_sharpe"]))
                if "test_sharpe" in rec:
                    mlflow.log_metric("test_sharpe", float(rec["test_sharpe"]))
                if "train_drawdown" in rec:
                    mlflow.log_metric("train_drawdown", float(rec["train_drawdown"]))
                if "test_drawdown" in rec:
                    mlflow.log_metric("test_drawdown", float(rec["test_drawdown"]))

        avg_test_equity = mean(r.get("test_equity", 0.0) for r in results)
        mlflow.log_metric("avg_test_equity", avg_test_equity)
        if any("test_sharpe" in r for r in results):
            avg_test_sharpe = mean(r.get("test_sharpe", 0.0) for r in results)
            mlflow.log_metric("avg_test_sharpe", avg_test_sharpe)
        if any("test_drawdown" in r for r in results):
            avg_test_drawdown = mean(r.get("test_drawdown", 0.0) for r in results)
            mlflow.log_metric("avg_test_drawdown", avg_test_drawdown)

    return {"results": results, "avg_test_equity": avg_test_equity}
