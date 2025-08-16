"""Event-driven backtesting engine with order queues and slippage models."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    exchange: str = field(default="default", compare=False)
    place_price: float = field(default=0.0, compare=False)
    remaining_qty: float = field(default=0.0, compare=False)
    filled_qty: float = field(default=0.0, compare=False)
    total_cost: float = field(default=0.0, compare=False)
    queue_pos: float = field(default=0.0, compare=False)


class SlippageModel:
    """Simple slippage model using volume impact and bid/ask spread.

    The fill price is adjusted by half the bar spread plus an additional
    component proportional to the order size divided by the bar volume.
    It also supports level-2 order book queue modelling by exposing
    :meth:`fill`, which returns the adjusted price, the filled quantity and
    the updated queue position.
    """

    def __init__(self, volume_impact: float = 0.1) -> None:
        self.volume_impact = float(volume_impact)

    def adjust(self, side: str, qty: float, price: float, bar: pd.Series) -> float:
        spread = float(bar["high"] - bar["low"])
        vol = float(bar.get("volume", 0.0))
        impact = self.volume_impact * qty / max(vol, 1e-9)
        slip = spread / 2 + impact
        return price + slip if side == "buy" else price - slip

    # ------------------------------------------------------------------
    def fill(
        self,
        side: str,
        qty: float,
        price: float,
        bar: pd.Series,
        queue_pos: float = 0.0,
        partial: bool = True,
    ) -> tuple[float, float, float]:
        """Return adjusted price, executed quantity and new queue position.

        Parameters
        ----------
        side: "buy" or "sell".
        qty: remaining order quantity.
        price: base execution price before slippage.
        bar: current market bar with optional L2 fields ``bid_size``/``ask_size``.
        queue_pos: volume ahead in the order book queue.
        partial: whether partial fills are allowed.
        """

        vol_key = "ask_size" if side == "buy" else "bid_size"
        avail = float(bar.get(vol_key, qty))
        eff_avail = max(0.0, avail - queue_pos)
        new_queue = max(0.0, queue_pos - avail)
        if partial:
            fill_qty = min(qty, eff_avail)
        else:
            fill_qty = qty if eff_avail >= qty else 0.0
        adj_price = self.adjust(side, fill_qty, price, bar)
        return adj_price, fill_qty, new_queue


class FeeModel:
    """Very small fee model applying a flat percentage to trade value."""

    def __init__(self, fee: float = 0.0) -> None:
        self.fee = float(fee)

    def calculate(self, cash: float) -> float:
        return abs(cash) * self.fee


class EventDrivenBacktestEngine:
    """Backtest engine supporting multiple symbols/strategies and order latency."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategies: Iterable[Tuple[str, str] | Tuple[str, str, str]],
        latency: int = 1,
        window: int = 120,
        slippage: SlippageModel | None = None,
        exchange_configs: Dict[str, Dict[str, float]] | None = None,
        use_l2: bool = False,
        partial_fills: bool = True,
        cancel_unfilled: bool = False,
    ) -> None:
        self.data = data
        self.latency = int(latency)
        self.window = int(window)
        self.slippage = slippage
        self.use_l2 = bool(use_l2)
        self.partial_fills = bool(partial_fills)
        self.cancel_unfilled = bool(cancel_unfilled)

        # Exchange specific configurations
        self.exchange_latency: Dict[str, int] = {}
        self.exchange_fees: Dict[str, FeeModel] = {}
        exchange_configs = exchange_configs or {}
        for exch, cfg in exchange_configs.items():
            self.exchange_latency[exch] = int(cfg.get("latency", latency))
            self.exchange_fees[exch] = FeeModel(cfg.get("fee", 0.0))
        self.default_fee = FeeModel(0.0)

        self.strategies: Dict[Tuple[str, str], object] = {}
        self.risk: Dict[Tuple[str, str], RiskManager] = {}
        self.strategy_exchange: Dict[Tuple[str, str], str] = {}
        for strat_info in strategies:
            if len(strat_info) == 2:
                strat_name, symbol = strat_info  # type: ignore[misc]
                exchange = "default"
            else:
                strat_name, symbol, exchange = strat_info  # type: ignore[misc]
            strat_cls = STRATEGIES.get(strat_name)
            if strat_cls is None:
                raise ValueError(f"unknown strategy: {strat_name}")
            key = (strat_name, symbol)
            self.strategies[key] = strat_cls()
            self.risk[key] = RiskManager(max_pos=1.0)
            self.strategy_exchange[key] = exchange

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Execute the backtest and return summary results."""

        equity = 0.0
        fills: List[tuple] = []
        order_queue: List[Order] = []
        orders: List[Order] = []
        slippage_total = 0.0
        funding_total = 0.0

        max_len = max(len(df) for df in self.data.values())
        for i in range(max_len):
            # Execute queued orders for this index
            while order_queue and order_queue[0].execute_index <= i:
                order = heapq.heappop(order_queue)
                df = self.data[order.symbol]
                if i >= len(df):
                    continue
                bar = df.iloc[i]
                if "bid" in bar and "ask" in bar:
                    price = float(bar["ask"] if order.side == "buy" else bar["bid"])
                else:
                    price = float(bar["close"])

                if self.slippage:
                    price, fill_qty, order.queue_pos = self.slippage.fill(
                        order.side,
                        order.remaining_qty,
                        price,
                        bar,
                        order.queue_pos,
                        self.partial_fills,
                    )
                else:
                    vol_key = "ask_size" if order.side == "buy" else "bid_size"
                    avail = float(bar.get(vol_key, order.remaining_qty))
                    if self.use_l2:
                        eff_avail = max(0.0, avail - order.queue_pos)
                        order.queue_pos = max(0.0, order.queue_pos - avail)
                    else:
                        eff_avail = avail
                        order.queue_pos = 0.0
                    if self.partial_fills:
                        fill_qty = min(order.remaining_qty, eff_avail)
                    else:
                        fill_qty = (
                            order.remaining_qty
                            if eff_avail >= order.remaining_qty
                            else 0.0
                        )

                if fill_qty <= 0:
                    if not self.cancel_unfilled:
                        order.execute_index = i + 1
                        heapq.heappush(order_queue, order)
                    continue

                slip = (
                    price - order.place_price
                    if order.side == "buy"
                    else order.place_price - price
                )
                slippage_total += slip * fill_qty

                cash = fill_qty * price
                fee_model = self.exchange_fees.get(order.exchange, self.default_fee)
                fee = fee_model.calculate(cash)
                if order.side == "buy":
                    equity -= cash + fee
                else:
                    equity += cash - fee
                self.risk[(order.strategy, order.symbol)].add_fill(
                    order.side, fill_qty
                )
                order.filled_qty += fill_qty
                order.remaining_qty -= fill_qty
                order.total_cost += price * fill_qty
                fills.append(
                    (
                        bar.get("timestamp", i),
                        price,
                        fill_qty,
                        order.strategy,
                        order.symbol,
                        order.exchange,
                    )
                )
                if order.remaining_qty > 1e-9 and not self.cancel_unfilled:
                    order.execute_index = i + 1
                    heapq.heappush(order_queue, order)

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
                exchange = self.strategy_exchange[(strat_name, symbol)]
                exec_index = i + self.exchange_latency.get(exchange, self.latency)
                place_price = float(df["close"].iloc[i])
                queue_pos = 0.0
                if self.use_l2:
                    vol_key = "ask_size" if side == "buy" else "bid_size"
                    queue_pos = float(df.iloc[i].get(vol_key, 0.0))
                order = Order(
                    exec_index,
                    strat_name,
                    symbol,
                    side,
                    qty,
                    exchange,
                    place_price,
                    qty,
                    0.0,
                    0.0,
                    queue_pos,
                )
                orders.append(order)
                heapq.heappush(order_queue, order)

            # ------------------------------------------------------------------
            # Apply funding payments after processing the bar
            for (strat_name, symbol), risk in self.risk.items():
                df = self.data[symbol]
                if i >= len(df):
                    continue
                bar = df.iloc[i]
                rate = float(bar.get("funding_rate", 0.0))
                if rate:
                    price = float(bar.get("close", 0.0))
                    pos = risk.pos.qty
                    cash = pos * price * rate
                    equity -= cash
                    funding_total += cash

        # Liquidate remaining positions
        for (strat_name, symbol), risk in self.risk.items():
            pos = risk.pos.qty
            if abs(pos) > 1e-9:
                last_price = float(self.data[symbol]["close"].iloc[-1])
                equity += pos * last_price

        orders_summary = [
            {
                "strategy": o.strategy,
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "filled": o.filled_qty,
                "place_price": o.place_price,
                "avg_price": o.total_cost / o.filled_qty if o.filled_qty else 0.0,
                "exchange": o.exchange,
            }
            for o in orders
        ]
        result = {
            "equity": equity,
            "fills": fills,
            "orders": orders_summary,
            "slippage": slippage_total,
            "funding": funding_total,
        }
        log.info("Backtest result: %s", result)
        return result


def run_backtest_csv(
    csv_paths: Dict[str, str],
    strategies: Iterable[Tuple[str, str]],
    latency: int = 1,
    window: int = 120,
    slippage: SlippageModel | None = None,
    exchange_configs: Dict[str, Dict[str, float]] | None = None,
    use_l2: bool = False,
    partial_fills: bool = True,
    cancel_unfilled: bool = False,
) -> dict:
    """Convenience wrapper to run the engine from CSV files."""

    data = {sym: pd.read_csv(Path(path)) for sym, path in csv_paths.items()}
    engine = EventDrivenBacktestEngine(
        data,
        strategies,
        latency=latency,
        window=window,
        slippage=slippage,
        exchange_configs=exchange_configs,
        use_l2=use_l2,
        partial_fills=partial_fills,
        cancel_unfilled=cancel_unfilled,
    )
    return engine.run()


def run_backtest_mlflow(
    csv_paths: Dict[str, str],
    strategies: Iterable[Tuple[str, str]],
    latency: int = 1,
    window: int = 120,
    slippage: SlippageModel | None = None,
    exchange_configs: Dict[str, Dict[str, float]] | None = None,
    use_l2: bool = False,
    partial_fills: bool = True,
    cancel_unfilled: bool = False,
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
            exchange_configs=exchange_configs,
            use_l2=use_l2,
            partial_fills=partial_fills,
            cancel_unfilled=cancel_unfilled,
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


