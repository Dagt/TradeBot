#!/usr/bin/env python3
"""Example CLI running walk-forward optimisation with MLflow logging."""

import argparse
import mlflow

from tradingbot.experiments.walk_forward import run_walk_forward_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="CSV file with historical OHLCV data")
    parser.add_argument("symbol", help="Trading symbol, e.g. BTC/USDT")
    parser.add_argument(
        "--experiment", default="walk-forward", help="MLflow experiment name"
    )
    parser.add_argument(
        "--mlflow-uri",
        default="./mlruns",
        help="MLflow tracking URI (directory for local runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)

    param_grid = [
        {"rsi_n": 10, "rsi_threshold": 55, "stop_loss_pct": 0.02},
        {"rsi_n": 14, "rsi_threshold": 60, "stop_loss_pct": 0.02},
    ]
    res = run_walk_forward_experiment(
        args.data,
        args.symbol,
        "momentum",
        param_grid,
        n_splits=3,
        experiment=args.experiment,
    )
    print(res)


if __name__ == "__main__":
    main()
