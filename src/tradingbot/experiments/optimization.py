"""Generic Optuna optimisation helpers."""
from __future__ import annotations

from typing import Any, Callable, Dict

import optuna

from .mlflow_utils import log_backtest_metrics, start_run


def optimize(
    param_space: Dict[str, Callable[[optuna.Trial], Any]],
    backtest_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    n_trials: int = 20,
    experiment: str = "optimization",
    direction: str = "maximize",
) -> optuna.study.Study:
    """Run a generic Optuna optimisation for a backtest function.

    Parameters
    ----------
    param_space:
        Mapping of parameter names to callables that accept an Optuna
        ``Trial`` and return the suggested value.
    backtest_fn:
        Callable executed on each trial receiving the sampled params and
        returning a result dictionary with an ``equity`` field and optional
        ``sharpe``.
    n_trials:
        Number of optimisation trials to execute.
    experiment:
        MLflow experiment name used to record each trial.
    direction:
        Direction of optimisation (``"maximize"`` or ``"minimize"``).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest(trial) for name, suggest in param_space.items()}
        with start_run(experiment, run_name=f"trial_{trial.number}", params=params):
            result = backtest_fn(params)
            log_backtest_metrics(result)
            return float(result.get("equity", 0.0))

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return study
