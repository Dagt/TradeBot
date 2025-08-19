"""Utilities for working with MLflow runs in backtests."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator

import mlflow


@contextmanager
def start_run(
    experiment: str,
    run_name: str | None = None,
    params: Dict[str, Any] | None = None,
    tags: Dict[str, str] | None = None,
    *,
    nested: bool = False,
) -> Iterator[None]:
    """Start an MLflow run under ``experiment`` and log optional params.

    Parameters
    ----------
    experiment:
        Name of the experiment to use (created if it does not exist).
    run_name:
        Optional run name displayed in the UI.
    params:
        Extra parameters to log once the run starts.
    tags:
        Optional tags for the run.
    nested:
        When ``True`` the run is started as a nested run under the currently
        active MLflow run.
    """

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested):
        if params:
            mlflow.log_params(params)
        yield


def log_backtest_metrics(result: Dict[str, Any]) -> None:
    """Log common backtest metrics to the active MLflow run."""

    mlflow.log_metric("equity", float(result.get("equity", 0.0)))
    mlflow.log_metric("fills", len(result.get("fills", [])))
    if "sharpe" in result:
        mlflow.log_metric("sharpe", float(result["sharpe"]))
    if "max_drawdown" in result:
        mlflow.log_metric("max_drawdown", float(result["max_drawdown"]))
