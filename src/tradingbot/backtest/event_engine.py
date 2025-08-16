import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import mlflow
import optuna

from ..strategies import STRATEGIES
from ..risk.manager import RiskManager

log = logging.getLogger(__name__)


def run_backtest_csv(
    csv_path: str,
    symbol: str = "BTC/USDT",
    strategy: str = "breakout_atr",
    strategy_params: Dict[str, Any] | None = None,
):
    path = Path(csv_path)
    df = pd.read_csv(path)
    # Expected columns: timestamp, open, high, low, close, volume
    equity = 0.0
    pos = 0.0
    strat_cls = STRATEGIES.get(strategy)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy}")
    strat = strat_cls(**(strategy_params or {}))
    risk = RiskManager(max_pos=1.0)

    window = 120  # bars for features
    fills = []

    for i in range(len(df)):
        if i < window:
            continue
        win = df.iloc[i-window:i].copy()
        bar = {"window": win}
        sig = strat.on_bar(bar)
        if sig is None:
            continue
        delta = risk.size(sig.side, sig.strength)
        # Fill at next close as simplificación
        px = df["close"].iloc[i]
        if abs(delta) > 1e-9:
            equity -= delta * px  # cash change
            pos += delta
            fills.append((df["timestamp"].iloc[i], px, delta))
    # Liquidar posición
    last_px = df["close"].iloc[-1]
    equity += pos * last_px
    result = {"equity": equity, "fills": fills, "last_px": last_px}
    log.info("Backtest result: %s", result)
    return result


def run_backtest_csv_mlflow(
    csv_path: str,
    symbol: str = "BTC/USDT",
    strategy: str = "breakout_atr",
    strategy_params: Dict[str, Any] | None = None,
    experiment: str = "backtests",
):
    """Run backtest and log metrics/params to MLflow."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{strategy}_{symbol}"):
        mlflow.log_params({"symbol": symbol, "strategy": strategy})
        if strategy_params:
            mlflow.log_params(strategy_params)
        res = run_backtest_csv(csv_path, symbol=symbol, strategy=strategy, strategy_params=strategy_params)
        mlflow.log_metric("equity", res["equity"])
        mlflow.log_metric("last_px", res["last_px"])
        mlflow.log_metric("fills", len(res["fills"]))
    return res


def optimize_strategy(
    csv_path: str,
    symbol: str = "BTC/USDT",
    strategy: str = "breakout_atr",
    n_trials: int = 10,
):
    """Optimize a strategy hyperparameter using Optuna.

    Currently optimizes ``atr_period`` for the ``breakout_atr`` strategy
    as a simple example.
    """

    def objective(trial: optuna.Trial) -> float:
        atr_period = trial.suggest_int("atr_period", 5, 50)
        params = {"atr_period": atr_period}
        res = run_backtest_csv(
            csv_path, symbol=symbol, strategy=strategy, strategy_params=params
        )
        return res["equity"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    log.info("Best params %s with equity %.2f", study.best_params, study.best_value)
    return study
