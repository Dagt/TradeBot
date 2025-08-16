"""Event-driven backtesting engine with order queues and slippage models."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import pandas as pd

from ..risk.manager import RiskManager
from ..strategies import STRATEGIES

log = logging.getLogger(__name__)


@dataclass(order=True)
class Order:
    """Representation of a pending order in the backtest queue."""

    execute_index: int
    strategy: str = field(compare=False)
    symbol: str = field(compare=False)
    side: str = field(compare=False)  # "buy" or "sell"
    qty: float = field(compare=False)


class SlippageModel:
    """Simple slippage model using volume impact and bid/ask spread.

    The fill price is adjusted by half the bar spread plus an additional
    component proportional to the order size divided by the bar volume.
    """

    def __init__(self, volume_impact: float = 0.1) -> None:
        self.volume_impact = float(volume_impact)

    def adjust(self, side: str, qty: float, price: float, bar: pd.Series) -> float:
        spread = float(bar["high"] - bar["low"])
        vol = float(bar.get("volume", 0.0))
        impact = self.volume_impact * qty / max(vol, 1e-9)
        slip = spread / 2 + impact
        return price + slip if side == "buy" else price - slip


class EventDrivenBacktestEngine:
    """Backtest engine supporting multiple symbols/strategies and order latency."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategies: Iterable[Tuple[str, str]],
        latency: int = 1,
        window: int = 120,
        slippage: SlippageModel | None = None,
    ) -> None:
        self.data = data
        self.latency = int(latency)
        self.window = int(window)
        self.slippage = slippage

        self.strategies: Dict[Tuple[str, str], object] = {}
        self.risk: Dict[Tuple[str, str], RiskManager] = {}
        for strat_name, symbol in strategies:
            strat_cls = STRATEGIES.get(strat_name)
            if strat_cls is None:
                raise ValueError(f"unknown strategy: {strat_name}")
            key = (strat_name, symbol)
            self.strategies[key] = strat_cls()
            self.risk[key] = RiskManager(max_pos=1.0)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Execute the backtest and return summary results."""

        equity = 0.0
        fills: List[tuple] = []
        order_queue: List[Order] = []

        max_len = max(len(df) for df in self.data.values())
        for i in range(max_len):
            # Execute queued orders for this index
            while order_queue and order_queue[0].execute_index <= i:
                order = heapq.heappop(order_queue)
                df = self.data[order.symbol]
                if i >= len(df):
                    continue
                bar = df.iloc[i]
                price = float(bar["close"])
                if self.slippage:
                    price = self.slippage.adjust(order.side, order.qty, price, bar)
                cash = order.qty * price
                if order.side == "buy":
                    equity -= cash
                else:
                    equity += cash
                self.risk[(order.strategy, order.symbol)].add_fill(order.side, order.qty)
                fills.append(
                    (bar.get("timestamp", i), price, order.qty, order.strategy, order.symbol)
                )

            # Generate new orders from strategies
            for (strat_name, symbol), strat in self.strategies.items():
                df = self.data[symbol]
                if i < self.window or i >= len(df):
                    continue
                window_df = df.iloc[i - self.window : i]
                sig = strat.on_bar({"window": window_df})
                if sig is None or sig.side == "flat":
                    continue
                delta = self.risk[(strat_name, symbol)].size(sig.side, sig.strength)
                if abs(delta) < 1e-9:
                    continue
                side = "buy" if delta > 0 else "sell"
                qty = abs(delta)
                exec_index = i + self.latency
                heapq.heappush(
                    order_queue,
                    Order(exec_index, strat_name, symbol, side, qty),
                )

        # Liquidate remaining positions
        for (strat_name, symbol), risk in self.risk.items():
            pos = risk.pos.qty
            if abs(pos) > 1e-9:
                last_price = float(self.data[symbol]["close"].iloc[-1])
                equity += pos * last_price

        result = {"equity": equity, "fills": fills}
        log.info("Backtest result: %s", result)
        return result


def run_backtest_csv(
    csv_paths: Dict[str, str],
    strategies: Iterable[Tuple[str, str]],
    latency: int = 1,
    window: int = 120,
    slippage: SlippageModel | None = None,
) -> dict:
    """Convenience wrapper to run the engine from CSV files."""

    data = {sym: pd.read_csv(Path(path)) for sym, path in csv_paths.items()}
    engine = EventDrivenBacktestEngine(
        data, strategies, latency=latency, window=window, slippage=slippage
    )
    return engine.run()


def run_backtest_mlflow(
    csv_paths: Dict[str, str],
    strategies: Iterable[Tuple[str, str]],
    latency: int = 1,
    window: int = 120,
    slippage: SlippageModel | None = None,
    run_name: str = "backtest",
    params: Dict[str, Any] | None = None,
) -> dict:
    """Run the backtest and log results to an MLflow run.

    Parameters
    ----------
    csv_paths: mapping of symbol to CSV path.
    strategies: iterable of ``(strategy_name, symbol)`` pairs.
    latency, window, slippage: same as :func:`run_backtest_csv`.
    run_name: optional MLflow run name.
    params: extra parameters to log to MLflow.
    """

    import mlflow  # type: ignore

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"latency": latency, "window": window})
        if params:
            mlflow.log_params(params)
        result = run_backtest_csv(
            csv_paths,
            strategies,
            latency=latency,
            window=window,
            slippage=slippage,
        )
        mlflow.log_metric("equity", result.get("equity", 0.0))
        mlflow.log_metric("fills", len(result.get("fills", [])))
    return result


def optimize_strategy_optuna(
    csv_path: str,
    symbol: str,
    strategy_name: str,
    param_space: Dict[str, Dict[str, Any]],
    n_trials: int = 20,
    latency: int = 1,
    window: int = 120,
) -> "optuna.study.Study":
    """Optimize strategy hyperparameters using Optuna.

    Parameters
    ----------
    csv_path: path to the CSV data for the symbol.
    symbol: market symbol (e.g. ``"BTC/USDT"``).
    strategy_name: name of the strategy present in ``STRATEGIES``.
    param_space: mapping of parameter names to Optuna suggestion
        dictionaries. Each value must specify ``type`` (``"int"`` or
        ``"float"``) and ``low``/``high`` bounds.
    n_trials: number of optimization trials.
    latency, window: engine parameters.
    """

    import optuna  # type: ignore

    df = pd.read_csv(Path(csv_path))
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")

    def objective(trial: "optuna.trial.Trial") -> float:
        params: Dict[str, Any] = {}
        for name, space in param_space.items():
            t = space.get("type")
            low, high = space.get("low"), space.get("high")
            if t == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif t == "float":
                params[name] = trial.suggest_float(name, float(low), float(high))
            else:
                raise ValueError(f"unsupported param type: {t}")

        strat = strat_cls(**params)
        engine = EventDrivenBacktestEngine(
            {symbol: df}, [(strategy_name, symbol)], latency=latency, window=window
        )
        engine.strategies[(strategy_name, symbol)] = strat
        res = engine.run()
        return float(res.get("equity", 0.0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


