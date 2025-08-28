"""Event-driven backtesting engine with order queues and slippage models."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections.abc import Mapping

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

# Minimum quantity to record a fill. Fills below this threshold are ignored.
MIN_FILL_QTY = 1e-6


def _validate_risk_pct(value: float) -> float:
    """Ensure ``risk_pct`` is a fraction in [0, 1].

    Values greater than 1 and up to 100 are interpreted as percentages and
    converted by dividing by 100. Values outside these ranges raise a
    :class:`ValueError`.
    """

    val = float(value)
    if val < 0:
        raise ValueError("risk_pct must be non-negative")
    if val >= 1:
        if val <= 100:
            return val / 100
        raise ValueError("risk_pct must be between 0 and 1")
    return val


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

    def adjust(
        self,
        side: str,
        qty: float,
        price: float,
        bar: Mapping[str, float] | pd.Series,
    ) -> float:
        spread = float(bar.get("high", 0.0) - bar.get("low", 0.0)) * self.spread_mult
        vol = float(bar.get("volume", 0.0))
        impact = self.volume_impact * qty / max(vol, 1e-9)
        ofi_val = 0.0
        if "order_flow_imbalance" in bar:
            ofi_val = float(bar["order_flow_imbalance"])
        elif {"bid_qty", "ask_qty"}.issubset(getattr(bar, "keys", lambda: [])()):
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
        bar: Mapping[str, float] | pd.Series,
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
    """Backtest engine supporting multiple symbols/strategies and order latency.

    Fills with quantity below ``min_fill_qty`` (defaults to
    ``MIN_FILL_QTY`` = ``1e-6``) are ignored to avoid recording irrelevant
    residuals.
    """

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
        initial_equity: float = 1000.0,
        risk_pct: float = 0.0,
        verbose_fills: bool = False,
        min_fill_qty: float = MIN_FILL_QTY,
    ) -> None:
        self.data = data
        self.latency = int(latency)
        self.window = int(window)
        self.slippage = slippage
        self.use_l2 = bool(use_l2)
        self.partial_fills = bool(partial_fills)
        self.cancel_unfilled = bool(cancel_unfilled)
        self.stress = stress or StressConfig()
        self.verbose_fills = bool(verbose_fills)
        self.min_fill_qty = float(min_fill_qty)

        # Set global random seeds for reproducibility
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Apply spread multiplier to slippage model if provided
        if self.slippage and hasattr(self.slippage, "spread_mult"):
            self.slippage.spread_mult *= self.stress.spread

        self.initial_equity = float(initial_equity)
        self._risk_pct = _validate_risk_pct(risk_pct)

        # Exchange specific configurations
        self.exchange_latency: Dict[str, int] = {}
        self.exchange_fees: Dict[str, FeeModel] = {}
        self.exchange_depth: Dict[str, float] = {}
        self.exchange_mode: Dict[str, str] = {}
        self.exchange_tick_size: Dict[str, float] = {}
        exchange_configs = exchange_configs or {}
        default_fee = 0.001
        for exch, cfg in exchange_configs.items():
            self.exchange_latency[exch] = int(cfg.get("latency", latency))
            self.exchange_fees[exch] = FeeModel(cfg.get("fee", default_fee))
            self.exchange_depth[exch] = float(cfg.get("depth", float("inf")))
            self.exchange_tick_size[exch] = float(cfg.get("tick_size", 0.0))
            market_type = cfg.get("market_type")
            if market_type is None:
                market_type = "spot" if exch.endswith("_spot") else "perp"
            self.exchange_mode[exch] = str(market_type)
        # Auto-derive market type for exchanges without explicit configs
        for strat_info in strategies:
            if len(strat_info) == 2:
                exchange = "default"
            else:
                exchange = strat_info[2]
            if exchange not in self.exchange_mode:
                self.exchange_mode[exchange] = (
                    "spot" if exchange.endswith("_spot") else "perp"
                )
        self.default_fee = FeeModel(default_fee)
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
            allow_short = self.exchange_mode.get(exchange, "perp") != "spot"
            rm = RiskManager(
                risk_pct=self._risk_pct,
                allow_short=allow_short,
            )
            guard = PortfolioGuard(GuardConfig(venue=exchange))
            self.risk[key] = RiskService(rm, guard)
            self.strategy_exchange[key] = exchange

    # ------------------------------------------------------------------
    def run(self, fills_csv: str | None = None) -> dict:
        """Execute the backtest and return summary results.

        Notes
        -----
        The reported Sharpe ratio is annualised assuming daily bars
        (252 trading days per year). If using a different bar frequency,
        adjust the square root factor accordingly.
        """
        max_len = max(len(df) for df in self.data.values())
        log.info(
            "Ejecutando backtest con %d estrategias y %d barras",
            len(self.strategies),
            max_len,
        )

        cash = self.initial_equity
        equity = cash
        collect_fills = bool(fills_csv or self.verbose_fills)
        fills: List[tuple] = [] if collect_fills else []
        order_queue: List[Order] = []
        orders: List[Order] = []
        slippage_total = 0.0
        funding_total = 0.0
        fill_count = 0
        equity_curve: List[float] = []
        data_arrays = {
            sym: {col: df[col].to_numpy() for col in df.columns}
            for sym, df in self.data.items()
        }
        data_lengths = {sym: len(df) for sym, df in self.data.items()}
        sym_vols = {
            sym: returns(df).rolling(self.window).std().to_numpy()
            for sym, df in self.data.items()
        }

        mtm = sum(
            svc.rm.pos.qty * data_arrays[sym]["close"][0]
            for (strat, sym), svc in self.risk.items()
            if data_lengths[sym] > 0
        )
        equity = cash + mtm
        equity_curve.append(equity)
        last_index = max_len - 1

        returns_df = pd.DataFrame(index=range(max_len)) if len(self.risk) > 1 else None
        rolling_corr = None
        rolling_cov = None
        if returns_df is not None:
            for sym, df_sym in self.data.items():
                returns_df[sym] = returns(df_sym).reset_index(drop=True)
            rolling_corr = returns_df.rolling(self.window).corr()
            rolling_cov = returns_df.rolling(self.window).cov()

        for i in range(max_len):
            if i and i % 500 == 0:
                log.info("Progreso: %d/%d barras", i, max_len)
            # Actualiza límites por correlación/covarianza con retornos recientes
            if (
                returns_df is not None
                and rolling_corr is not None
                and rolling_cov is not None
                and i >= self.window
            ):
                corr_df = rolling_corr.loc[i]
                cov_df = rolling_cov.loc[i]
                for svc in self.risk.values():
                    svc.update_correlation(corr_df, 0.8)
                    svc.update_covariance(cov_df, 0.8)
            # Execute queued orders for this index
            while order_queue and order_queue[0].execute_index <= i:
                order = heapq.heappop(order_queue)
                arrs = data_arrays[order.symbol]
                sym_len = data_lengths[order.symbol]
                if i >= sym_len:
                    continue
                vol_key = "ask_size" if order.side == "buy" else "bid_size"
                vol_arr = arrs.get(vol_key)
                avail = float(vol_arr[i]) if vol_arr is not None else order.remaining_qty
                if self.use_l2:
                    depth = self.exchange_depth.get(order.exchange, self.default_depth)
                    order.queue_pos = min(order.queue_pos + avail, depth)
                if "bid" in arrs and "ask" in arrs:
                    price = float(arrs["ask"][i] if order.side == "buy" else arrs["bid"][i])
                else:
                    price = float(arrs["close"][i])

                svc = self.risk[(order.strategy, order.symbol)]
                mode = self.exchange_mode.get(order.exchange, "perp")

                if self.slippage:
                    bar = {vol_key: avail}
                    for k in (
                        "high",
                        "low",
                        "volume",
                        "order_flow_imbalance",
                        "bid_qty",
                        "ask_qty",
                    ):
                        arr_k = arrs.get(k)
                        if arr_k is not None:
                            bar[k] = float(arr_k[i])
                    price, fill_qty, order.queue_pos = self.slippage.fill(
                        order.side,
                        order.remaining_qty,
                        price,
                        bar,
                        order.queue_pos,
                        self.partial_fills,
                    )
                else:
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

                if mode == "spot":
                    if order.side == "sell":
                        avail_base = max(0.0, svc.rm.pos.qty)
                        fill_qty = min(fill_qty, avail_base)
                    else:
                        fee_model_tmp = self.exchange_fees.get(order.exchange, self.default_fee)
                        max_affordable = cash / (price * (1 + fee_model_tmp.fee))
                        fill_qty = min(fill_qty, max_affordable)

                if fill_qty <= 0 or fill_qty < self.min_fill_qty:
                    if not self.cancel_unfilled:
                        order.execute_index = i + 1
                        heapq.heappush(order_queue, order)
                    continue

                fill_qty = round(fill_qty, 8)

                fee_model = self.exchange_fees.get(order.exchange, self.default_fee)
                trade_value = fill_qty * price
                fee = fee_model.calculate(trade_value)

                if mode == "spot":
                    if order.side == "sell":
                        while svc.rm.pos.qty - fill_qty < 0 and fill_qty > 0:
                            fill_qty = round(fill_qty - 1e-8, 8)
                            trade_value = fill_qty * price
                            fee = fee_model.calculate(trade_value)
                    else:
                        while cash - (trade_value + fee) < 0 and fill_qty > 0:
                            fill_qty = round(fill_qty - 1e-8, 8)
                            trade_value = fill_qty * price
                            fee = fee_model.calculate(trade_value)

                if fill_qty <= 0 or fill_qty < self.min_fill_qty:
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
                fill_count += 1

                trade_value = fill_qty * price
                fee = fee_model.calculate(trade_value)
                if order.side == "buy":
                    cash -= trade_value + fee
                else:
                    cash += trade_value - fee
                svc.on_fill(order.symbol, order.side, fill_qty, price)
                order.filled_qty += fill_qty
                order.remaining_qty -= fill_qty
                order.total_cost += price * fill_qty
                if order.remaining_qty <= 1e-9:
                    if order.latency is None:
                        order.latency = i - order.place_index
                    svc.rm.complete_order()

                base_after = svc.rm.pos.qty
                mtm_after = 0.0
                for (strat_s, sym_s), svc_s in self.risk.items():
                    arrs_s = data_arrays[sym_s]
                    sym_len_s = data_lengths[sym_s]
                    idx_px = i if i < sym_len_s else sym_len_s - 1
                    mark = price if sym_s == order.symbol else float(arrs_s["close"][idx_px])
                    mtm_after += svc_s.rm.pos.qty * mark
                equity_after = cash + mtm_after

                timestamp = arrs.get("timestamp")[i] if "timestamp" in arrs else i
                if collect_fills:
                    fills.append(
                        (
                            timestamp,
                            order.side,
                            price,
                            fill_qty,
                            order.strategy,
                            order.symbol,
                            order.exchange,
                            fee,
                            cash,
                            base_after,
                            equity_after,
                            getattr(svc.rm.pos, "realized_pnl", 0.0),
                        )
                    )
                if self.verbose_fills and not fills_csv:
                    log.info(
                        "Fill %s %s side=%s qty=%s price=%.2f rpnl=%.2f",
                        order.strategy,
                        order.symbol,
                        order.side,
                        f"{fill_qty:.8f}",
                        price,
                        getattr(svc.rm.pos, "realized_pnl", 0.0),
                    )
                if order.remaining_qty > 1e-9 and not self.cancel_unfilled:
                    order.execute_index = i + 1
                    heapq.heappush(order_queue, order)

            mtm = sum(
                svc.rm.pos.qty * data_arrays[sym]["close"][i]
                for (strat, sym), svc in self.risk.items()
                if i < data_lengths[sym]
            )
            equity = cash + mtm
            if equity < 0:
                log.warning(
                    "Equity depleted at bar %d; stopping backtest", i
                )
                equity_curve.append(equity)
                last_index = i
                break

            # Generate new orders from strategies
            for (strat_name, symbol), strat in self.strategies.items():
                df = self.data[symbol]
                arrs = data_arrays[symbol]
                sym_len = data_lengths[symbol]
                if i < self.window or i >= sym_len:
                    continue
                start_idx = i - self.window
                if getattr(strat, "needs_window_df", True):
                    window_df = df.iloc[start_idx:i]
                    sig = strat.on_bar({"window": window_df})
                else:
                    bar_arrays = {col: arrs[col][start_idx:i] for col in arrs}
                    sig = strat.on_bar(bar_arrays)
                if sig is None or sig.side == "flat":
                    continue
                svc = self.risk[(strat_name, symbol)]
                place_price = float(arrs["close"][i])
                svc.mark_price(symbol, place_price)
                symbol_vol = float(sym_vols[symbol][i])
                if np.isnan(symbol_vol):
                    symbol_vol = 0.0
                if equity < 0:
                    continue
                equity_for_order = max(equity, 1.0)
                allowed, _reason, delta = svc.check_order(
                    symbol,
                    sig.side,
                    equity_for_order,
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
                    vol_arr = arrs.get(vol_key)
                    avail = float(vol_arr[i]) if vol_arr is not None else 0.0
                    queue_pos = min(avail, depth)
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
                arrs = data_arrays[symbol]
                sym_len = data_lengths[symbol]
                if i >= sym_len:
                    continue
                rate_arr = arrs.get("funding_rate")
                rate = float(rate_arr[i]) if rate_arr is not None else 0.0
                if rate:
                    price = float(arrs["close"][i])
                    pos = svc.rm.pos.qty
                    funding_cash = pos * price * rate
                    cash -= funding_cash
                    funding_total += funding_cash

            # Re-check risk limits after funding adjustments
            for (strat_name, symbol), svc in self.risk.items():
                arrs = data_arrays[symbol]
                sym_len = data_lengths[symbol]
                if i >= sym_len:
                    continue
                current_price = float(arrs["close"][i])
                svc.mark_price(symbol, current_price)
                try:
                    svc.rm.check_limits(current_price)
                except StopLossExceeded:
                    delta = -svc.rm.pos.qty
                    if abs(delta) > 1e-9:
                        side = "buy" if delta > 0 else "sell"
                        qty = abs(delta)
                        exchange = self.strategy_exchange[(strat_name, symbol)]
                        base_latency = self.exchange_latency.get(exchange, self.latency)
                        exec_index = i + int(base_latency * self.stress.latency)
                        order = Order(
                            exec_index,
                            i,
                            strat_name,
                            symbol,
                            side,
                            qty,
                            exchange,
                            current_price,
                            qty,
                            0.0,
                            0.0,
                            0.0,
                        )
                        orders.append(order)
                        heapq.heappush(order_queue, order)

            # Track equity after processing each bar
            mtm = sum(
                svc.rm.pos.qty * data_arrays[sym]["close"][i]
                for (strat, sym), svc in self.risk.items()
                if i < data_lengths[sym]
            )
            equity = cash + mtm
            equity_curve.append(equity)

            if equity <= 0 and i > 0 and not order_queue:
                log.warning(
                    "Equity depleted at bar %d; stopping backtest", i
                )
                last_index = i
                break

            if i == max_len - 1:
                last_index = i
                break

        # Liquidate remaining positions
        for (strat_name, symbol), svc in self.risk.items():
            pos = svc.rm.pos.qty
            if abs(pos) > 1e-9:
                arrs = data_arrays[symbol]
                price_idx = min(last_index, data_lengths[symbol] - 1)
                last_price = float(arrs["close"][price_idx])
                cash += pos * last_price
                svc.rm.set_position(0.0)

        equity = cash
        # Update final equity in the curve without duplicating the last value
        equity_curve[-1] = equity

        equity_series = pd.Series(equity_curve)
        # Compute annualised Sharpe ratio from equity changes assuming daily bars
        rets = equity_series.diff().dropna()
        sharpe = (
            float((rets.mean() / rets.std()) * np.sqrt(252))
            if not rets.empty and rets.std()
            else 0.0
        )
        # Maximum drawdown from the equity curve
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max.clip(lower=1e-9)
        max_drawdown = -float(drawdown.min()) if not drawdown.empty else 0.0
        max_drawdown = min(max_drawdown, 1.0)

        pnl = equity_curve[-1] - self.initial_equity

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
            "equity": equity_curve[-1],
            "pnl": pnl,
            "orders": orders_summary,
            "slippage": slippage_total,
            "funding": funding_total,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "equity_curve": equity_curve,
            "fill_count": fill_count,
        }
        if collect_fills:
            result["fills"] = fills
        log.info(
            "Backtest finalizado: equity %.2f, pnl %.2f, fills %d, drawdown %.2f%%",
            result["equity"],
            result["pnl"],
            fill_count,
            result["max_drawdown"] * 100,
        )
        if fills_csv:
            fills_df = pd.DataFrame(
                result["fills"],
                columns=[
                    "timestamp",
                    "side",
                    "price",
                    "qty",
                    "strategy",
                    "symbol",
                    "exchange",
                    "fee",
                    "cash_after",
                    "base_after",
                    "equity_after",
                    "realized_pnl",
                ],
            )
            fills_df.to_csv(fills_csv, index=False)
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
    initial_equity: float = 1000.0,
    verbose_fills: bool = False,
    fills_csv: str | None = None,
) -> dict:
    """Convenience wrapper to run the engine from CSV files."""

    data = {sym: pd.read_csv(Path(path)) for sym, path in csv_paths.items()}
    if fills_csv:
        verbose_fills = False
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
        verbose_fills=verbose_fills,
    )
    return engine.run(fills_csv=fills_csv)


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
    initial_equity: float = 1000.0,
    verbose_fills: bool = False,
    fills_csv: str | None = None,
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
            verbose_fills=verbose_fills,
            fills_csv=fills_csv,
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


