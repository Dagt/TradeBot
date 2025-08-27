"""Event-driven backtesting engine with order queues and slippage models."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import random

from ..risk.manager import RiskManager
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.service import RiskService
from ..risk.exceptions import StopLossExceeded
from ..strategies import STRATEGIES
from ..data.features import returns, calc_ofi

log = logging.getLogger(__name__)


@dataclass(order=True)
class Order:
    """Representation of a pending order in the backtest queue."""

    execute_index: int
    place_index: int = field(compare=False)
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
    latency: int | None = field(default=None, compare=False)


class SlippageModel:
    """Simple slippage model using volume impact and bid/ask spread.

    The fill price is adjusted by half the bar spread plus an additional
    component proportional to the order size divided by the bar volume.
    It also supports level-2 order book queue modelling by exposing
    :meth:`fill`, which returns the adjusted price, the filled quantity and
    the updated queue position.  A ``spread_mult`` parameter allows stressing
    the bid/ask spread.
    """

    def __init__(
        self,
        volume_impact: float = 0.1,
        spread_mult: float = 1.0,
        ofi_impact: float = 0.0,
    ) -> None:
        self.volume_impact = float(volume_impact)
        self.spread_mult = float(spread_mult)
        self.ofi_impact = float(ofi_impact)

    def adjust(self, side: str, qty: float, price: float, bar: pd.Series) -> float:
        spread = float(bar["high"] - bar["low"]) * self.spread_mult
        vol = float(bar.get("volume", 0.0))
        impact = self.volume_impact * qty / max(vol, 1e-9)
        ofi_val = 0.0
        if "order_flow_imbalance" in bar:
            ofi_val = float(bar["order_flow_imbalance"])
        elif {"bid_qty", "ask_qty"}.issubset(bar.index):
            ofi_val = float(
                calc_ofi(
                    pd.DataFrame([[bar["bid_qty"], bar["ask_qty"]]], columns=["bid_qty", "ask_qty"])
                ).iloc[-1]
            )
        slip = spread / 2 + impact + self.ofi_impact * ofi_val
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


@dataclass
class StressConfig:
    """Multipliers to apply stress scenarios to the engine."""

    latency: float = 1.0
    spread: float = 1.0


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
        stress: StressConfig | None = None,
        seed: int | None = None,
        initial_equity: float = 0.0,
        risk_pct: float = 0.0,
    ) -> None:
        self.data = data
        self.latency = int(latency)
        self.window = int(window)
        self.slippage = slippage
        self.use_l2 = bool(use_l2)
        self.partial_fills = bool(partial_fills)
        self.cancel_unfilled = bool(cancel_unfilled)
        self.stress = stress or StressConfig()

        # Set global random seeds for reproducibility
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Apply spread multiplier to slippage model if provided
        if self.slippage and hasattr(self.slippage, "spread_mult"):
            self.slippage.spread_mult *= self.stress.spread

        self.initial_equity = float(initial_equity)
        self._risk_pct = float(risk_pct)

        # Exchange specific configurations
        self.exchange_latency: Dict[str, int] = {}
        self.exchange_fees: Dict[str, FeeModel] = {}
        self.exchange_depth: Dict[str, float] = {}
        exchange_configs = exchange_configs or {}
        for exch, cfg in exchange_configs.items():
            self.exchange_latency[exch] = int(cfg.get("latency", latency))
            self.exchange_fees[exch] = FeeModel(cfg.get("fee", 0.0))
            self.exchange_depth[exch] = float(cfg.get("depth", float("inf")))
        self.default_fee = FeeModel(0.0)
        self.default_depth = float("inf")

        self.strategies: Dict[Tuple[str, str], object] = {}
        self.risk: Dict[Tuple[str, str], RiskService] = {}
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
            rm = RiskManager(
                risk_pct=self._risk_pct,
            )
            guard = PortfolioGuard(GuardConfig(venue=exchange))
            self.risk[key] = RiskService(rm, guard)
            self.strategy_exchange[key] = exchange

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Execute the backtest and return summary results."""
        max_len = max(len(df) for df in self.data.values())
        log.info(
            "Ejecutando backtest con %d estrategias y %d barras",
            len(self.strategies),
            max_len,
        )

        equity = self.initial_equity
        fills: List[tuple] = []
        order_queue: List[Order] = []
        orders: List[Order] = []
        slippage_total = 0.0
        funding_total = 0.0
        equity_curve: List[float] = []
        equity_curve.append(equity)

        for i in range(max_len):
            if i and i % 1000 == 0:
                log.info("Progreso: %d/%d barras", i, max_len)
            # Actualiza límites por correlación/covarianza con retornos recientes
            if i >= self.window:
                returns_dict: Dict[str, List[float]] = {}
                for sym, df_sym in self.data.items():
                    window_df_sym = df_sym.iloc[i - self.window : i]
                    rets = returns(window_df_sym).dropna().tolist()
                    if rets:
                        returns_dict[sym] = rets
                if returns_dict:
                    corr_pairs: Dict[tuple[str, str], float] = {}
                    if len(returns_dict) > 1:
                        rets_df = pd.DataFrame(returns_dict)
                        cols = list(rets_df.columns)
                        for a_idx in range(len(cols)):
                            for b_idx in range(a_idx + 1, len(cols)):
                                a = cols[a_idx]
                                b = cols[b_idx]
                                corr = rets_df[a].corr(rets_df[b])
                                if not pd.isna(corr):
                                    corr_pairs[(a, b)] = float(corr)
                    any_rm = next(iter(self.risk.values()))
                    cov_matrix = any_rm.rm.covariance_matrix(returns_dict)
                    for svc in self.risk.values():
                        if corr_pairs:
                            svc.rm.update_correlation(corr_pairs, 0.8)
                        svc.rm.update_covariance(cov_matrix, 0.8)
            # Execute queued orders for this index
            while order_queue and order_queue[0].execute_index <= i:
                order = heapq.heappop(order_queue)
                df = self.data[order.symbol]
                if i >= len(df):
                    continue
                bar = df.iloc[i]
                vol_key = "ask_size" if order.side == "buy" else "bid_size"
                if self.use_l2:
                    depth = self.exchange_depth.get(order.exchange, self.default_depth)
                    order.queue_pos = min(
                        order.queue_pos + float(bar.get(vol_key, 0.0)), depth
                    )
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
                svc = self.risk[(order.strategy, order.symbol)]
                svc.on_fill(order.symbol, order.side, fill_qty)
                order.filled_qty += fill_qty
                order.remaining_qty -= fill_qty
                order.total_cost += price * fill_qty
                if order.remaining_qty <= 1e-9:
                    if order.latency is None:
                        order.latency = i - order.place_index
                    svc.rm.complete_order()
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

            # Update marks and enforce risk limits for open positions
            for (strat_name, symbol), svc in self.risk.items():
                df = self.data[symbol]
                if i >= len(df):
                    continue
                price_now = float(df["close"].iloc[i])
                svc.mark_price(symbol, price_now)
                try:
                    svc.rm.check_limits(price_now)
                except StopLossExceeded:
                    delta = -svc.rm.pos.qty
                    if abs(delta) > 1e-9:
                        side = "buy" if delta > 0 else "sell"
                        qty = abs(delta)
                        exchange = self.strategy_exchange[(strat_name, symbol)]
                        base_latency = self.exchange_latency.get(
                            exchange, self.latency
                        )
                        exec_index = i + int(base_latency * self.stress.latency)
                        order = Order(
                            exec_index,
                            i,
                            strat_name,
                            symbol,
                            side,
                            qty,
                            exchange,
                            price_now,
                            qty,
                            0.0,
                            0.0,
                            0.0,
                        )
                        orders.append(order)
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
                svc = self.risk[(strat_name, symbol)]
                place_price = float(df["close"].iloc[i])
                svc.mark_price(symbol, place_price)
                rets = returns(window_df).dropna()
                symbol_vol = float(rets.std()) if not rets.empty else 0.0
                eq_for_size = equity if equity > 0 else 100.0
                allowed, _reason, delta = svc.check_order(
                    symbol,
                    sig.side,
                    eq_for_size,
                    place_price,
                    strength=sig.strength,
                    symbol_vol=symbol_vol,
                    corr_threshold=0.8,
                )
                if not allowed or abs(delta) < 1e-9:
                    continue
                side = "buy" if delta > 0 else "sell"
                qty = abs(delta)
                notional = qty * place_price
                if not svc.rm.register_order(notional):
                    continue
                exchange = self.strategy_exchange[(strat_name, symbol)]
                base_latency = self.exchange_latency.get(exchange, self.latency)
                exec_index = i + int(base_latency * self.stress.latency)
                queue_pos = 0.0
                if self.use_l2:
                    vol_key = "ask_size" if side == "buy" else "bid_size"
                    depth = self.exchange_depth.get(exchange, self.default_depth)
                    queue_pos = min(float(df.iloc[i].get(vol_key, 0.0)), depth)
                order = Order(
                    exec_index,
                    i,
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
            for (strat_name, symbol), svc in self.risk.items():
                df = self.data[symbol]
                if i >= len(df):
                    continue
                bar = df.iloc[i]
                rate = float(bar.get("funding_rate", 0.0))
                if rate:
                    price = float(bar.get("close", 0.0))
                    pos = svc.rm.pos.qty
                    cash = pos * price * rate
                    equity -= cash
                    funding_total += cash

            # Track equity after processing each bar
            equity_curve.append(equity)

        # Liquidate remaining positions
        for (strat_name, symbol), svc in self.risk.items():
            pos = svc.rm.pos.qty
            if abs(pos) > 1e-9:
                last_price = float(self.data[symbol]["close"].iloc[-1])
                equity += pos * last_price

        # Final equity value
        equity_curve.append(equity)

        equity_series = pd.Series(equity_curve)
        # Compute simple Sharpe ratio from equity changes
        rets = equity_series.diff().dropna()
        sharpe = float(rets.mean() / rets.std()) if not rets.empty and rets.std() else 0.0
        # Maximum drawdown from the equity curve
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = -float(drawdown.min()) if not drawdown.empty else 0.0

        pnl = equity - self.initial_equity

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
                "latency": o.latency,
            }
            for o in orders
        ]

        result = {
            "equity": equity,
            "pnl": pnl,
            "fills": fills,
            "orders": orders_summary,
            "slippage": slippage_total,
            "funding": funding_total,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "equity_curve": equity_curve,
        }
        log.info(
            "Backtest finalizado: equity %.2f, pnl %.2f, fills %d, drawdown %.2f%%",
            result["equity"],
            result["pnl"],
            len(result["fills"]),
            result["max_drawdown"] * 100,
        )
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
    stress: StressConfig | None = None,
    seed: int | None = None,
    risk_pct: float = 0.0,
    initial_equity: float = 0.0,
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
        stress=stress,
        seed=seed,
        risk_pct=risk_pct,
        initial_equity=initial_equity,
    )
    return engine.run()


from ..experiments.mlflow_utils import log_backtest_metrics, start_run


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
    stress: StressConfig | None = None,
    seed: int | None = None,
    experiment: str = "backtest",
    risk_pct: float = 0.0,
    initial_equity: float = 0.0,
) -> dict:
    """Run the backtest and log results to an MLflow run.

    Parameters
    ----------
    csv_paths: mapping of symbol to CSV path.
    strategies: iterable of ``(strategy_name, symbol)`` pairs.
    latency, window, slippage: same as :func:`run_backtest_csv`.
    run_name: optional MLflow run name.
    params: extra parameters to log to MLflow.
    experiment: MLflow experiment name.
    """

    param_log = {"latency": latency, "window": window}
    if params:
        param_log.update(params)
    with start_run(experiment, run_name=run_name, params=param_log):
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
            stress=stress,
            seed=seed,
            risk_pct=risk_pct,
            initial_equity=initial_equity,
        )
        log_backtest_metrics(result)
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


def purged_kfold(
    n_samples: int, n_splits: int, embargo: float = 0.0
) -> List[Tuple[List[int], List[int]]]:
    """Generate K-fold partitions with an embargo to avoid look-ahead bias.

    Parameters
    ----------
    n_samples: int
        Total number of observations to split.
    n_splits: int
        Number of folds.
    embargo: float, optional
        Fraction of ``n_samples`` to exclude on each side of the test split
        from the training set.  For example ``embargo=0.01`` will remove 1 % of
        samples before and after the test window.

    Returns
    -------
    list of tuple
        Each tuple contains the training indices and the testing indices for
        the fold.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    indices = list(range(n_samples))
    fold_sizes = [n_samples // n_splits] * n_splits
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1

    embargo_size = int(n_samples * embargo)
    current = 0
    folds: List[Tuple[List[int], List[int]]] = []
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
        folds.append((train_idx, test_idx))
        current = stop
    return folds


def walk_forward_optimize(
    csv_path: str,
    symbol: str,
    strategy_name: str,
    param_grid: List[Dict[str, Any]],
    train_size: int | None = None,
    test_size: int | None = None,
    latency: int = 1,
    window: int = 120,
    *,
    splits: List[Tuple[List[int], List[int]]] | None = None,
    n_splits: int | None = None,
    embargo: float = 0.0,
) -> List[Dict[str, Any]]:
    """Perform a simple walk-forward optimization.

    Two modes are supported.  By default the function performs sequential
    walk-forward splits using ``train_size`` and ``test_size`` windows.  When
    ``splits`` or ``n_splits`` is provided the optimisation runs on the
    partitions returned by :func:`purged_kfold`, applying the specified
    ``embargo`` fraction to avoid look-ahead bias.
    """

    df = pd.read_csv(Path(csv_path))
    strat_cls = STRATEGIES.get(strategy_name)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy_name}")

    if splits is None and n_splits is not None:
        splits = purged_kfold(len(df), n_splits, embargo)

    results: List[Dict[str, Any]] = []

    if splits is not None:
        for split, (train_idx, test_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            best_params: Dict[str, Any] | None = None
            best_equity = -float("inf")
            best_res: Dict[str, Any] | None = None
            for params in param_grid:
                strat = strat_cls(**params)
                engine = EventDrivenBacktestEngine(
                    {symbol: train_df},
                    [(strategy_name, symbol)],
                    latency=latency,
                    window=window,
                )
                engine.strategies[(strategy_name, symbol)] = strat
                res = engine.run()
                eq = float(res.get("equity", 0.0))
                if eq > best_equity:
                    best_equity = eq
                    best_params = params
                    best_res = res

            strat = strat_cls(**(best_params or {}))
            engine = EventDrivenBacktestEngine(
                {symbol: test_df}, [(strategy_name, symbol)], latency=latency, window=window
            )
            engine.strategies[(strategy_name, symbol)] = strat
            test_res = engine.run()

            rec = {
                "split": split,
                "params": best_params,
                "train_equity": best_equity,
                "test_equity": float(test_res.get("equity", 0.0)),
            }
            if best_res is not None:
                rec["train_sharpe"] = float(best_res.get("sharpe", 0.0))
                rec["train_drawdown"] = float(best_res.get("max_drawdown", 0.0))
            rec["test_sharpe"] = float(test_res.get("sharpe", 0.0))
            rec["test_drawdown"] = float(test_res.get("max_drawdown", 0.0))
            log.info("walk-forward split %s: %s", split, rec)
            results.append(rec)
        return results

    if train_size is None or test_size is None:
        raise ValueError("train_size and test_size must be provided when splits is None")

    start = 0
    split = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start : start + train_size].reset_index(drop=True)
        test_df = df.iloc[start + train_size : start + train_size + test_size].reset_index(drop=True)

        best_params: Dict[str, Any] | None = None
        best_equity = -float("inf")
        best_res: Dict[str, Any] | None = None
        for params in param_grid:
            strat = strat_cls(**params)
            engine = EventDrivenBacktestEngine(
                {symbol: train_df}, [(strategy_name, symbol)], latency=latency, window=window
            )
            engine.strategies[(strategy_name, symbol)] = strat
            res = engine.run()
            eq = float(res.get("equity", 0.0))
            if eq > best_equity:
                best_equity = eq
                best_params = params
                best_res = res

        strat = strat_cls(**(best_params or {}))
        engine = EventDrivenBacktestEngine(
            {symbol: test_df}, [(strategy_name, symbol)], latency=latency, window=window
        )
        engine.strategies[(strategy_name, symbol)] = strat
        test_res = engine.run()

        rec = {
            "split": split,
            "params": best_params,
            "train_equity": best_equity,
            "test_equity": float(test_res.get("equity", 0.0)),
        }
        if best_res is not None:
            rec["train_sharpe"] = float(best_res.get("sharpe", 0.0))
            rec["train_drawdown"] = float(best_res.get("max_drawdown", 0.0))
        rec["test_sharpe"] = float(test_res.get("sharpe", 0.0))
        rec["test_drawdown"] = float(test_res.get("max_drawdown", 0.0))
        log.info("walk-forward split %s: %s", split, rec)
        results.append(rec)

        start += test_size
        split += 1

    return results


