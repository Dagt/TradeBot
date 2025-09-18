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
import math
from numba import njit

from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.service import RiskService
from ..core.account import Account as CoreAccount
from ..risk.exceptions import StopLossExceeded
from ..strategies import STRATEGIES
from ..data.features import returns, calc_ofi

log = logging.getLogger(__name__)

# Minimum quantity to record a fill. Fills below this threshold are ignored.
MIN_FILL_QTY = 0.0
# Minimum absolute order quantity. Orders below this are discarded.
MIN_ORDER_QTY = 1e-9


@njit(cache=True)
def _fee_calc(cash: float, rate: float) -> float:
    return abs(cash) * rate


@njit(cache=True)
def _slippage_core(
    side_is_buy: bool,
    qty: float,
    price: float,
    spread: float,
    volume: float,
    ofi_val: float,
    volume_impact: float,
    ofi_impact: float,
    pct: float,
) -> float:
    impact = volume_impact * qty / max(volume, 1e-9)
    slip = spread / 2 + impact + ofi_impact * ofi_val + price * pct
    return price + slip if side_is_buy else price - slip


@njit(cache=True)
def _fill_core(
    side_is_buy: bool,
    qty: float,
    price: float,
    spread: float,
    volume: float,
    avail: float,
    queue_pos: float,
    volume_impact: float,
    ofi_impact: float,
    ofi_val: float,
    pct: float,
    partial: bool,
) -> tuple[float, float, float]:
    eff_avail = max(0.0, avail - queue_pos)
    new_queue = max(0.0, queue_pos - avail)
    if partial:
        fill_qty = qty if qty < eff_avail else eff_avail
    else:
        fill_qty = qty if eff_avail >= qty else 0.0
    adj_price = _slippage_core(
        side_is_buy,
        fill_qty,
        price,
        spread,
        volume,
        ofi_val,
        volume_impact,
        ofi_impact,
        pct,
    )
    return adj_price, fill_qty, new_queue


def _validate_risk_pct(value: float) -> float:
    """Ensure ``risk_pct`` is a fraction in [0, 1].

    Values from 1 to 100 are interpreted as percentages and converted by
    dividing by 100. For example, a value of ``1`` becomes ``0.01`` (1 %).
    Values outside ``0``–``100`` raise a :class:`ValueError`.
    """

    val = float(value)
    if val < 0:
        raise ValueError("risk_pct must be non-negative")
    if val > 100:
        raise ValueError("risk_pct must be between 0 and 100")
    if val >= 1:
        return val / 100
    return val


@dataclass(order=True)
class Order:
    """Representation of a pending order in the backtest queue."""

    execute_index: int
    order_id: int = field(compare=False)
    place_index: int = field(compare=False)
    strategy: str = field(compare=False)
    symbol: str = field(compare=False)
    side: str = field(compare=False)  # "buy" or "sell"
    qty: float = field(compare=False)
    exchange: str = field(default="default", compare=False)
    place_price: float = field(default=0.0, compare=False)
    limit_price: float | None = field(default=None, compare=False)
    remaining_qty: float = field(default=0.0, compare=False)
    filled_qty: float = field(default=0.0, compare=False)
    total_cost: float = field(default=0.0, compare=False)
    queue_pos: float = field(default=0.0, compare=False)
    latency: int | None = field(default=None, compare=False)
    post_only: bool = field(default=False, compare=False)
    trailing_pct: float | None = field(default=None, compare=False)


class SlippageModel:
    """Simple slippage model using volume impact and bid/ask spread.

    The fill price is adjusted by half the bar spread plus an additional
    component proportional to the order size divided by the bar volume.
    It also supports level-2 order book queue modelling by exposing
    :meth:`fill`, which returns the adjusted price, the filled quantity and
    the updated queue position.  A ``spread_mult`` parameter allows stressing
    the bid/ask spread.

    Parameters
    ----------
    volume_impact: float, default ``0.1``
        Impact factor relative to bar volume.
    spread_mult: float, default ``1.0``
        Multiplier applied to the computed spread.
    ofi_impact: float, default ``0.0``
        Weight applied to the order flow imbalance term.
    source: {"bba", "fixed_spread"}, default ``"bba"``
        Source for the base spread. ``"bba"`` uses best bid/ask columns when
        available and falls back to ``base_spread`` when they are missing or
        contain ``NaN`` values. ``"fixed_spread"`` always uses ``base_spread``.
    base_spread: float, default ``0.0``
        Fallback spread (absolute price difference) used when bid/ask columns
        are absent, ``NaN`` or when ``source`` is ``"fixed_spread"``.
    """

    def __init__(
        self,
        volume_impact: float = 0.1,
        spread_mult: float = 1.0,
        ofi_impact: float = 0.0,
        source: str = "bba",
        base_spread: float = 0.0,
        pct: float = 0.0001,
    ) -> None:
        self.volume_impact = float(volume_impact)
        self.spread_mult = float(spread_mult)
        self.ofi_impact = float(ofi_impact)
        source = source.lower()
        if source not in {"bba", "fixed_spread"}:
            raise ValueError("source must be 'bba' or 'fixed_spread'")
        self.source = source
        self.base_spread = float(base_spread)
        self.pct = float(pct)
    def _compute_spread(self, bar: Mapping[str, float] | pd.Series) -> float:
        if self.source == "bba":
            bid = bar.get("bid") or bar.get("bid_px") or bar.get("bid_price")
            ask = bar.get("ask") or bar.get("ask_px") or bar.get("ask_price")
            if bid is not None and ask is not None:
                bid = float(bid)
                ask = float(ask)
                if not (math.isnan(bid) or math.isnan(ask)):
                    spread = ask - bid
                else:
                    spread = self.base_spread
            else:
                spread = self.base_spread
        else:
            spread = self.base_spread
        return spread * self.spread_mult

    def _ofi_from_bar(self, bar: Mapping[str, float] | pd.Series) -> float:
        if "order_flow_imbalance" in bar:
            return float(bar["order_flow_imbalance"])
        if {"bid_qty", "ask_qty"}.issubset(getattr(bar, "keys", lambda: [])()):
            return float(
                calc_ofi(
                    pd.DataFrame(
                        [[bar["bid_qty"], bar["ask_qty"]]],
                        columns=["bid_qty", "ask_qty"],
                    )
                ).iloc[-1]
            )
        return 0.0

    def adjust(
        self,
        side: str,
        qty: float,
        price: float,
        bar: Mapping[str, float] | pd.Series,
    ) -> float:
        spread = self._compute_spread(bar)
        vol = float(bar.get("volume", 0.0))
        ofi_val = self._ofi_from_bar(bar)
        side_is_buy = side == "buy"
        return float(
            _slippage_core(
                side_is_buy,
                qty,
                price,
                spread,
                vol,
                ofi_val,
                self.volume_impact,
                self.ofi_impact,
                self.pct,
            )
        )

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
        spread = self._compute_spread(bar)
        vol = float(bar.get("volume", 0.0))
        ofi_val = self._ofi_from_bar(bar)
        vol_key = "ask_size" if side == "buy" else "bid_size"
        avail = float(bar.get(vol_key, qty))
        side_is_buy = side == "buy"
        adj_price, fill_qty, new_queue = _fill_core(
            side_is_buy,
            qty,
            price,
            spread,
            vol,
            avail,
            queue_pos,
            self.volume_impact,
            self.ofi_impact,
            ofi_val,
            self.pct,
            partial,
        )
        return float(adj_price), float(fill_qty), float(new_queue)


class FeeModel:
    """Fee model with distinct maker/taker percentages in basis points."""

    def __init__(
        self, maker_fee_bps: float = 0.0, taker_fee_bps: float | None = None
    ) -> None:
        self.maker_fee = float(maker_fee_bps) / 10000.0
        taker = maker_fee_bps if taker_fee_bps is None else taker_fee_bps
        self.taker_fee = float(taker) / 10000.0

    def calculate(self, cash: float, maker: bool = False) -> float:
        rate = self.maker_fee if maker else self.taker_fee
        return float(_fee_calc(cash, rate))


@dataclass
class StressConfig:
    """Multipliers to apply stress scenarios to the engine."""

    latency: float = 1.0
    spread: float = 1.0


class EventDrivenBacktestEngine:
    """Backtest engine supporting multiple symbols/strategies and order latency.

    Fills with quantity below ``min_fill_qty`` (defaults to
    ``MIN_FILL_QTY`` = ``0.0``) are ignored to avoid recording irrelevant
    residuals.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategies: Iterable[Tuple[str, str] | Tuple[str, str, str]],
        latency: int = 1,
        window: int = 120,
        slippage: SlippageModel | None = None,
        fee_bps: float = 10.0,
        slippage_bps: float = 0.0,
        exchange_configs: Dict[str, Dict[str, float]] | None = None,
        use_l2: bool = False,
        partial_fills: bool = True,
        cancel_unfilled: bool = False,
        stress: StressConfig | None = None,
        seed: int | None = None,
        initial_equity: float = 1000.0,
        risk_pct: float = 0.0,
        risk_per_trade: float = 1.0,
        verbose_fills: bool = False,
        min_fill_qty: float = MIN_FILL_QTY,
        min_order_qty: float = MIN_ORDER_QTY,
        fast: bool = False,
    ) -> None:
        self.data = data
        self.latency = int(latency)
        self.window = int(window)
        self.slippage = slippage
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        if self.slippage is None:
            self.slippage = SlippageModel(
                volume_impact=0.0, pct=self.slippage_bps / 10000.0
            )
        else:
            self.slippage.pct = self.slippage_bps / 10000.0
        self.use_l2 = bool(use_l2)
        self.partial_fills = bool(partial_fills)
        self.cancel_unfilled = bool(cancel_unfilled)
        self.stress = stress or StressConfig()
        self.verbose_fills = bool(verbose_fills)
        self.min_fill_qty = float(min_fill_qty)
        self.min_order_qty = float(min_order_qty)
        self.fast = bool(fast)

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
        self.risk_per_trade = float(risk_per_trade)

        # Exchange specific configurations
        self.exchange_latency: Dict[str, int] = {}
        self.exchange_fees: Dict[str, FeeModel] = {}
        self.exchange_depth: Dict[str, float] = {}
        self.exchange_mode: Dict[str, str] = {}
        self.exchange_tick_size: Dict[str, float] = {}
        self.exchange_min_fill_qty: Dict[str, float] = {}
        exchange_configs = exchange_configs or {}
        default_maker_bps = self.fee_bps
        default_taker_bps = self.fee_bps
        for exch, cfg in exchange_configs.items():
            self.exchange_latency[exch] = int(cfg.get("latency", latency))
            maker_bps = float(cfg.get("maker_fee_bps", default_maker_bps))
            taker_bps = float(
                cfg.get("taker_fee_bps", cfg.get("maker_fee_bps", default_taker_bps))
            )
            self.exchange_fees[exch] = FeeModel(maker_bps, taker_bps)
            self.exchange_depth[exch] = float(cfg.get("depth", float("inf")))
            self.exchange_tick_size[exch] = float(cfg.get("tick_size", 0.0))
            self.exchange_min_fill_qty[exch] = float(cfg.get("min_fill_qty", 0.0))
            market_type = cfg.get("market_type")
            if market_type is None:
                if exch.endswith("_spot"):
                    market_type = "spot"
                elif exch.endswith("_futures") or exch.endswith("_perp"):
                    market_type = "perp"
                else:
                    # Default to perpetual markets when the type cannot be inferred
                    market_type = "perp"
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
        self.default_fee = FeeModel(default_maker_bps, default_taker_bps)
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
            market_mode = self.exchange_mode.get(exchange, "perp")
            mt = "spot" if market_mode == "spot" else "futures"
            guard = PortfolioGuard(GuardConfig(venue=exchange))
            account = CoreAccount(float("inf"), cash=self.initial_equity, market_type=mt)
            svc = RiskService(
                guard,
                account=account,
                risk_per_trade=self.risk_per_trade,
                risk_pct=self._risk_pct,
                market_type=mt,
            )
            svc.min_order_qty = self.min_order_qty
            self.risk[key] = svc
            self.strategy_exchange[key] = exchange
            try:
                strat = strat_cls(risk_service=svc)
            except TypeError:
                strat = strat_cls()
                setattr(strat, "risk_service", svc)
            self.strategies[key] = strat

        # Internal flag to avoid repeated on_bar calls per bar index
        self._last_on_bar_i: int = -1

    # ------------------------------------------------------------------
    def run(self, fills_csv: str | None = None) -> dict:
        """Execute the backtest and return summary results.

        Volatility per symbol is no longer precomputed; any strategy or risk
        component requiring it should derive the necessary metrics from price
        data directly.

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

        def sync_cash() -> None:
            for svc in self.risk.values():
                svc.account.cash = cash

        sync_cash()
        collect_fills = bool(fills_csv or self.verbose_fills)
        fills: List[tuple] = [] if collect_fills else []
        order_queue: List[Order] = []
        orders: List[Order] = []
        position_levels: Dict[Tuple[str, str], Dict[str, Any]] = {}
        slippage_total = 0.0
        funding_total = 0.0
        fill_count = 0
        realized_pnl_total = 0.0
        equity_curve: List[float] = []
        data_arrays = {
            sym: {col: df[col].to_numpy() for col in df.columns}
            for sym, df in self.data.items()
        }
        data_lengths = {sym: len(df) for sym, df in self.data.items()}

        order_seq = 0

        mtm = sum(
            svc.account.current_exposure(sym)[0] * data_arrays[sym]["close"][0]
            for (strat, sym), svc in self.risk.items()
            if data_lengths[sym] > 0
        )
        equity = cash + mtm
        equity_curve.append(equity)
        last_index = max_len - 1

        returns_df = (
            pd.DataFrame(index=range(max_len))
            if len(self.risk) > 1 and not self.fast
            else None
        )
        rolling_corr = None
        rolling_cov = None
        if returns_df is not None:
            for sym, df_sym in self.data.items():
                returns_df[sym] = returns(df_sym).reset_index(drop=True)
            rolling_corr = returns_df.rolling(self.window).corr()
            rolling_cov = returns_df.rolling(self.window).cov()

        self._last_on_bar_i = -1

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
            for (strat, sym), svc in self.risk.items():
                arrs = data_arrays[sym]
                sym_len = data_lengths[sym]
                if i < sym_len:
                    price = float(arrs["close"][i])
                    pos_qty, _ = svc.account.current_exposure(sym)
                    trade = svc.get_trade(sym)
                    if trade and abs(pos_qty) > self.min_order_qty:
                        svc.update_trailing(trade, price)
                        trade["_trail_done"] = True
                        trade["current_price"] = price
                        decision = svc.manage_position(trade)
                        if decision in {"scale_in", "scale_out"}:
                            target = svc.calc_position_size(
                                trade.get("strength", 1.0), price, clamp=False
                            )
                            delta_qty = target - abs(pos_qty)
                            if abs(delta_qty) > self.min_order_qty:
                                side = (
                                    trade["side"]
                                    if delta_qty > 0
                                    else ("sell" if trade["side"] == "buy" else "buy")
                                )
                                exchange = self.strategy_exchange[(strat, sym)]
                                base_latency = self.exchange_latency.get(
                                    exchange, self.latency
                                )
                                delay = max(1, int(base_latency * self.stress.latency))
                                exec_index = i + delay
                                queue_pos = 0.0
                                if self.use_l2:
                                    vol_key = (
                                        "ask_size" if side == "buy" else "bid_size"
                                    )
                                    depth = self.exchange_depth.get(
                                        exchange, self.default_depth
                                    )
                                    vol_arr = arrs.get(vol_key)
                                    avail = (
                                        float(vol_arr[i])
                                        if vol_arr is not None
                                        else 0.0
                                    )
                                    queue_pos = min(avail, depth)
                                svc.account.update_open_order(sym, side, abs(delta_qty))
                                order_seq += 1
                                order = Order(
                                    exec_index,
                                    order_seq,
                                    i,
                                    strat,
                                    sym,
                                    side,
                                    abs(delta_qty),
                                    exchange,
                                    price,
                                    price,
                                    abs(delta_qty),
                                    0.0,
                                    0.0,
                                    queue_pos,
                                    None,
                                    False,
                                    None,
                                )
                                orders.append(order)
                                heapq.heappush(order_queue, order)
                        elif decision == "close":
                            delta_qty = -pos_qty
                            pending_qty = abs(delta_qty)
                            if pending_qty > self.min_order_qty:
                                side = "sell" if pos_qty > 0 else "buy"
                                exchange = self.strategy_exchange[(strat, sym)]
                                base_latency = self.exchange_latency.get(
                                    exchange, self.latency
                                )
                                delay = max(1, int(base_latency * self.stress.latency))
                                exec_index = i + delay
                                queue_pos = 0.0
                                if self.use_l2:
                                    vol_key = (
                                        "ask_size" if side == "buy" else "bid_size"
                                    )
                                    depth = self.exchange_depth.get(
                                        exchange, self.default_depth
                                    )
                                    vol_arr = arrs.get(vol_key)
                                    avail = (
                                        float(vol_arr[i])
                                        if vol_arr is not None
                                        else 0.0
                                    )
                                    queue_pos = min(avail, depth)
                                svc.account.update_open_order(sym, side, pending_qty)
                                order_seq += 1
                                order = Order(
                                    exec_index,
                                    order_seq,
                                    i,
                                    strat,
                                    sym,
                                    side,
                                    pending_qty,
                                    exchange,
                                    price,
                                    price,
                                    pending_qty,
                                    0.0,
                                    0.0,
                                    queue_pos,
                                    None,
                                    False,
                                    None,
                                )
                                orders.append(order)
                                heapq.heappush(order_queue, order)
            # Execute queued orders for this index
            while order_queue and order_queue[0].execute_index <= i:
                order = heapq.heappop(order_queue)
                arrs = data_arrays[order.symbol]
                sym_len = data_lengths[order.symbol]
                if i >= sym_len:
                    continue
                vol_key = "ask_size" if order.side == "buy" else "bid_size"
                vol_arr = arrs.get(vol_key)
                if self.use_l2:
                    avail = float(vol_arr[i]) if vol_arr is not None else 0.0
                    depth = self.exchange_depth.get(order.exchange, self.default_depth)
                    order.queue_pos = min(order.queue_pos + avail, depth)
                else:
                    avail = order.remaining_qty
                    order.queue_pos = 0.0
                if "bid" in arrs and "ask" in arrs:
                    market_price = float(
                        arrs["ask"][i] if order.side == "buy" else arrs["bid"][i]
                    )
                else:
                    market_price = float(arrs["close"][i])
                price = market_price
                if order.limit_price is not None:
                    if order.side == "buy":
                        price = min(market_price, order.limit_price)
                    else:
                        price = max(market_price, order.limit_price)

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

                if order.limit_price is not None:
                    if order.side == "buy":
                        price = min(price, order.limit_price)
                    else:
                        price = max(price, order.limit_price)

                if not self.fast and mode == "spot":
                    if order.side == "sell":
                        avail_base = max(
                            0.0, svc.account.current_exposure(order.symbol)[0]
                        )
                        fill_qty = min(fill_qty, avail_base)
                    else:
                        fee_model_tmp = self.exchange_fees.get(
                            order.exchange, self.default_fee
                        )
                        rate = (
                            fee_model_tmp.maker_fee
                            if order.post_only
                            else fee_model_tmp.taker_fee
                        )
                        max_affordable = cash / (price * (1 + rate))
                        fill_qty = min(fill_qty, max_affordable)

                min_fill_qty = self.exchange_min_fill_qty.get(
                    order.exchange, self.min_fill_qty
                )
                if fill_qty <= 0 or fill_qty < min_fill_qty:
                    if not self.cancel_unfilled:
                        order.execute_index = i + 1
                        heapq.heappush(order_queue, order)
                    continue

                fill_qty = round(fill_qty, 8)

                fee_model = self.exchange_fees.get(order.exchange, self.default_fee)
                maker = order.post_only
                trade_value = fill_qty * price
                fee_cost = fee_model.calculate(trade_value, maker=maker)

                if not self.fast and mode == "spot":
                    if order.side == "sell":
                        while (
                            svc.account.current_exposure(order.symbol)[0] - fill_qty < 0
                            and fill_qty > 0
                        ):
                            fill_qty = round(fill_qty - 1e-8, 8)
                            trade_value = fill_qty * price
                            fee_cost = fee_model.calculate(trade_value, maker=maker)
                    else:
                        while cash - (trade_value + fee_cost) < 0 and fill_qty > 0:
                            fill_qty = round(fill_qty - 1e-8, 8)
                            trade_value = fill_qty * price
                            fee_cost = fee_model.calculate(trade_value, maker=maker)

                if fill_qty <= 0 or fill_qty < min_fill_qty:
                    if not self.cancel_unfilled:
                        order.execute_index = i + 1
                        heapq.heappush(order_queue, order)
                    continue

                slip = (
                    price - order.place_price
                    if order.side == "buy"
                    else order.place_price - price
                )
                slip_cash = slip * fill_qty
                slippage_total += slip_cash
                fill_count += 1

                trade_value = fill_qty * price
                fee_cost = fee_model.calculate(trade_value, maker=maker)
                prev_rpnl = getattr(svc.pos, "realized_pnl", 0.0)
                if order.side == "buy":
                    cash -= trade_value + fee_cost
                else:
                    cash += trade_value - fee_cost
                sync_cash()
                prev_qty = svc.account.current_exposure(order.symbol)[0]
                svc.on_fill(order.symbol, order.side, fill_qty, price)
                new_rpnl = getattr(svc.pos, "realized_pnl", 0.0)
                realized_pnl = new_rpnl - prev_rpnl - fee_cost - slip_cash
                slippage_pnl = -slip_cash
                realized_pnl_total += realized_pnl
                svc.pos.realized_pnl = prev_rpnl + realized_pnl
                new_qty = svc.account.current_exposure(order.symbol)[0]
                key = (order.strategy, order.symbol)
                prev_sign = 1 if prev_qty > 0 else -1 if prev_qty < 0 else 0
                new_sign = 1 if new_qty > 0 else -1 if new_qty < 0 else 0
                if new_sign == 0:
                    position_levels.pop(key, None)
                elif prev_sign == 0 or prev_sign != new_sign:
                    position_levels[key] = {
                        "entry_i": i,
                        "trail_pct": order.trailing_pct,
                        "best_price": price,
                    }
                else:
                    state = position_levels.get(key)
                    if state is not None:
                        state["entry_i"] = i
                        if order.trailing_pct is not None:
                            state["trail_pct"] = order.trailing_pct
                        if state.get("trail_pct") is not None:
                            if new_sign > 0:
                                state["best_price"] = max(
                                    state.get("best_price", price), price
                                )
                            else:
                                state["best_price"] = min(
                                    state.get("best_price", price), price
                                )
                if not self.fast and mode == "spot":
                    if cash < -1e-9:
                        raise AssertionError(
                            f"cash became negative after {order.side} {order.symbol}: {cash}"
                        )
                    if svc.account.current_exposure(order.symbol)[0] < -1e-9:
                        raise AssertionError(
                            f"position went negative for {order.symbol}: {svc.account.current_exposure(order.symbol)[0]}"
                        )
                order.filled_qty += fill_qty
                order.remaining_qty -= fill_qty
                order.total_cost += price * fill_qty
                if order.remaining_qty <= 1e-9:
                    if order.latency is None:
                        order.latency = i - order.place_index
                    svc.complete_order(
                        venue=order.exchange,
                        symbol=order.symbol,
                        side=order.side,
                    )

                mtm_after = 0.0
                for (strat_s, sym_s), svc_s in self.risk.items():
                    arrs_s = data_arrays[sym_s]
                    sym_len_s = data_lengths[sym_s]
                    idx_px = i if i < sym_len_s else sym_len_s - 1
                    mark = (
                        price
                        if sym_s == order.symbol
                        else float(arrs_s["close"][idx_px])
                    )
                    mtm_after += svc_s.account.current_exposure(sym_s)[0] * mark
                equity_after = cash + mtm_after

                timestamp = arrs.get("timestamp")[i] if "timestamp" in arrs else i
                if collect_fills:
                    fills.append(
                        (
                            timestamp,
                            "order",
                            order.side,
                            price,
                            fill_qty,
                            order.strategy,
                            order.symbol,
                            order.exchange,
                            fee_cost,
                            slippage_pnl,
                            realized_pnl,
                            realized_pnl_total,
                            equity_after,
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
                        getattr(svc.pos, "realized_pnl", 0.0),
                    )
                if order.remaining_qty > 1e-9 and not self.cancel_unfilled:
                    order.execute_index = i + 1
                    heapq.heappush(order_queue, order)

            # ------------------------------------------------------------------
            # After executing pending orders, evaluate TP/SL/Trailing exits
            for (strat_name, symbol), state in list(position_levels.items()):
                svc = self.risk[(strat_name, symbol)]
                arrs = data_arrays[symbol]
                sym_len = data_lengths[symbol]
                if i >= sym_len:
                    continue
                qty = svc.account.current_exposure(symbol)[0]
                if abs(qty) < self.min_order_qty:
                    continue
                if state.get("entry_i") == i:
                    continue
                high = float(arrs.get("high", arrs["close"])[i])
                low = float(arrs.get("low", arrs["close"])[i])
                exit_price = None
                side = None
                trail = state.get("trail_pct")
                best = state.get("best_price", float(arrs["close"][i]))
                reason = ""
                if qty > 0 and trail is not None:
                    best = max(best, high)
                    state["best_price"] = best
                    tr_stop = best * (1 - trail)
                    if low <= tr_stop:
                        exit_price = tr_stop
                        side = "sell"
                        reason = "trailing_stop"
                elif qty < 0 and trail is not None:
                    best = min(best, low)
                    state["best_price"] = best
                    tr_stop = best * (1 + trail)
                    if high >= tr_stop:
                        exit_price = tr_stop
                        side = "buy"
                        reason = "trailing_stop"
                if exit_price is not None:
                    exit_qty = abs(qty)
                    exchange = self.strategy_exchange[(strat_name, symbol)]
                    fee_model = self.exchange_fees.get(exchange, self.default_fee)
                    trade_value = exit_qty * exit_price
                    fee_cost = fee_model.calculate(trade_value, maker=False)
                    prev_rpnl = getattr(svc.pos, "realized_pnl", 0.0)
                    if side == "sell":
                        cash += trade_value - fee_cost
                    else:
                        cash -= trade_value + fee_cost
                    sync_cash()
                    prev_qty = svc.account.current_exposure(symbol)[0]
                    svc.on_fill(symbol, side, exit_qty, exit_price)
                    new_rpnl = getattr(svc.pos, "realized_pnl", 0.0)
                    realized_pnl = new_rpnl - prev_rpnl - fee_cost
                    slippage_pnl = 0.0
                    realized_pnl_total += realized_pnl
                    svc.pos.realized_pnl = prev_rpnl + realized_pnl
                    position_levels.pop((strat_name, symbol), None)
                    mtm_after = 0.0
                    for (strat_s, sym_s), svc_s in self.risk.items():
                        arrs_s = data_arrays[sym_s]
                        sym_len_s = data_lengths[sym_s]
                        idx_px = i if i < sym_len_s else sym_len_s - 1
                        mark = (
                            exit_price
                            if sym_s == symbol
                            else float(arrs_s["close"][idx_px])
                        )
                        mtm_after += svc_s.account.current_exposure(sym_s)[0] * mark
                    equity_after = cash + mtm_after
                    if collect_fills:
                        timestamp = (
                            arrs.get("timestamp")[i] if "timestamp" in arrs else i
                        )
                        fills.append(
                            (
                                timestamp,
                                reason or "exit",
                                side,
                                exit_price,
                                exit_qty,
                                strat_name,
                                symbol,
                                exchange,
                                fee_cost,
                                slippage_pnl,
                                realized_pnl,
                                realized_pnl_total,
                                equity_after,
                            )
                        )

            # After executing pending orders and intrabar exits, update equity/positions
            mtm = sum(
                svc.account.current_exposure(sym)[0] * data_arrays[sym]["close"][i]
                for (strat, sym), svc in self.risk.items()
                if i < data_lengths[sym]
            )
            equity = cash + mtm
            if equity < 0:
                log.warning("Equity depleted at bar %d; stopping backtest", i)
                equity_curve.append(equity)
                last_index = i
                break

            # ------------------------------------------------------------------
            # Run each strategy's on_bar once per bar index
            if self._last_on_bar_i != i:
                self._last_on_bar_i = i
                # Generate new orders from strategies
                for (strat_name, symbol), strat in self.strategies.items():
                    df = self.data[symbol]
                    arrs = data_arrays[symbol]
                    sym_len = data_lengths[symbol]
                    if i < self.window or i >= sym_len:
                        continue
                    svc = self.risk[(strat_name, symbol)]
                    pos_qty, _ = svc.account.current_exposure(symbol)
                    trade = svc.get_trade(symbol)
                    start_idx = i - self.window
                    if getattr(strat, "needs_window_df", True):
                        window_df = df.iloc[start_idx:i]
                        sig = strat.on_bar({"window": window_df, "symbol": symbol})
                    else:
                        bar_arrays = {col: arrs[col][start_idx:i] for col in arrs}
                        bar_arrays["symbol"] = symbol
                        sig = strat.on_bar(bar_arrays)
                    limit_price = (
                        sig.get("limit_price")
                        if isinstance(sig, dict)
                        else getattr(sig, "limit_price", None)
                    )
                    if limit_price is None:
                        place_price = float(arrs["close"][i])
                        limit_price = place_price
                    else:
                        place_price = float(limit_price)
                        limit_price = place_price
                    svc.mark_price(symbol, place_price)
                    if equity < 0:
                        continue
                    if trade:
                        trade["current_price"] = place_price
                        sig_obj = sig.__dict__ if hasattr(sig, "__dict__") else sig
                        decision = svc.manage_position(trade, sig_obj)
                        limit_price = (
                            sig.get("limit_price")
                            if isinstance(sig, dict)
                            else getattr(sig, "limit_price", None)
                        )
                        if limit_price is None:
                            place_price = float(arrs["close"][i])
                            limit_price = place_price
                        else:
                            place_price = float(limit_price)
                            limit_price = place_price
                        svc.mark_price(symbol, place_price)
                        if decision == "close":
                            delta_qty = -pos_qty
                        elif decision in {"scale_in", "scale_out"}:
                            target = svc.calc_position_size(sig.strength, place_price, clamp=False)
                            delta_qty = target - abs(pos_qty)
                        else:
                            continue
                        if abs(delta_qty) < self.min_order_qty:
                            continue
                        if decision == "close":
                            side = "buy" if delta_qty > 0 else "sell"
                        else:
                            side = (
                                trade["side"]
                                if delta_qty > 0
                                else ("sell" if trade["side"] == "buy" else "buy")
                            )
                        qty = abs(delta_qty)
                        notional = qty * place_price
                        if not svc.register_order(symbol, notional):
                            continue
                        svc.account.update_open_order(symbol, side, qty)
                        exchange = self.strategy_exchange[(strat_name, symbol)]
                        base_latency = self.exchange_latency.get(exchange, self.latency)
                        delay = max(1, int(base_latency * self.stress.latency))
                        exec_index = i + delay
                        queue_pos = 0.0
                        if self.use_l2:
                            vol_key = "ask_size" if side == "buy" else "bid_size"
                            depth = self.exchange_depth.get(
                                exchange, self.default_depth
                            )
                            vol_arr = arrs.get(vol_key)
                            avail = float(vol_arr[i]) if vol_arr is not None else 0.0
                            queue_pos = min(avail, depth)
                        post_only = bool(getattr(sig, "post_only", False))
                        order_seq += 1
                        order = Order(
                            exec_index,
                            order_seq,
                            i,
                            strat_name,
                            symbol,
                            side,
                            qty,
                            exchange,
                            place_price,
                            limit_price,
                            qty,
                            0.0,
                            0.0,
                            queue_pos,
                            None,
                            post_only,
                            None,
                        )
                        orders.append(order)
                        heapq.heappush(order_queue, order)
                        continue
                    if sig is None or sig.side == "flat":
                        continue
                    if (
                        sig.side == "sell"
                        and not svc.allow_short
                        and svc.account.current_exposure(symbol)[0] <= 0
                    ):
                        continue
                    pending = svc.account.open_orders.get(symbol, {}).get(sig.side, 0.0)
                    atr_map = getattr(strat, "_last_atr", {})
                    tgt_map = getattr(strat, "_last_target_vol", {})
                    allowed, _reason, delta = svc.check_order(
                        symbol,
                        sig.side,
                        place_price,
                        strength=sig.strength,
                        pending_qty=pending,
                        volatility=atr_map.get(symbol),
                        target_volatility=tgt_map.get(symbol),
                    )
                    if not allowed or abs(delta) < self.min_order_qty:
                        continue
                    side = "buy" if delta > 0 else "sell"
                    qty = abs(delta)
                    notional = qty * place_price
                    if not svc.register_order(symbol, notional):
                        continue
                    svc.account.update_open_order(symbol, side, qty)
                    exchange = self.strategy_exchange[(strat_name, symbol)]
                    base_latency = self.exchange_latency.get(exchange, self.latency)
                    delay = max(1, int(base_latency * self.stress.latency))
                    exec_index = i + delay
                    queue_pos = 0.0
                    if self.use_l2:
                        vol_key = "ask_size" if side == "buy" else "bid_size"
                        depth = self.exchange_depth.get(exchange, self.default_depth)
                        vol_arr = arrs.get(vol_key)
                        avail = float(vol_arr[i]) if vol_arr is not None else 0.0
                        queue_pos = min(avail, depth)
                    post_only = bool(getattr(sig, "post_only", False))
                    order_seq += 1
                    order = Order(
                        exec_index,
                        order_seq,
                        i,
                        strat_name,
                        symbol,
                        side,
                        qty,
                        exchange,
                        place_price,
                        limit_price,
                        qty,
                        0.0,
                        0.0,
                        queue_pos,
                        None,
                        post_only,
                        None,
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
                    pos = svc.account.current_exposure(symbol)[0]
                    funding_cash = pos * price * rate
                    cash -= funding_cash
                    funding_total += funding_cash
                    sync_cash()

            # Re-check risk limits after funding adjustments
            if not self.fast:
                for (strat_name, symbol), svc in self.risk.items():
                    arrs = data_arrays[symbol]
                    sym_len = data_lengths[symbol]
                    if i >= sym_len:
                        continue
                    current_price = float(arrs["close"][i])
                    svc.mark_price(symbol, current_price)
                    check_price = current_price
                    pos_qty, _ = svc.account.current_exposure(symbol)
                    if pos_qty > 0 and "low" in arrs:
                        check_price = float(arrs["low"][i])
                    elif pos_qty < 0 and "high" in arrs:
                        check_price = float(arrs["high"][i])
                    try:
                        svc.check_limits(check_price)
                    except StopLossExceeded:
                        delta = -svc.account.current_exposure(symbol)[0]
                        if abs(delta) > self.min_order_qty:
                            side = "buy" if delta > 0 else "sell"
                            qty = abs(delta)
                            exchange = self.strategy_exchange[(strat_name, symbol)]
                            base_latency = self.exchange_latency.get(exchange, self.latency)
                            delay = max(1, int(base_latency * self.stress.latency))
                            exec_index = i + delay
                            order_seq += 1
                            order = Order(
                                exec_index,
                                order_seq,
                                i,
                                strat_name,
                                symbol,
                                side,
                                qty,
                                exchange,
                                current_price,
                                None,
                                qty,
                                0.0,
                                0.0,
                                0.0,
                                None,
                                False,
                                None,
                            )
                            orders.append(order)
                            heapq.heappush(order_queue, order)

            # Track equity after processing each bar
            mtm = sum(
                svc.account.current_exposure(sym)[0] * data_arrays[sym]["close"][i]
                for (strat, sym), svc in self.risk.items()
                if i < data_lengths[sym]
            )
            equity = cash + mtm
            equity_curve.append(equity)

            if equity <= 0 and i > 0 and not order_queue:
                log.warning("Equity depleted at bar %d; stopping backtest", i)
                last_index = i
                break

            if i == max_len - 1:
                last_index = i
                break

        # Liquidate remaining positions
        for (strat_name, symbol), svc in self.risk.items():
            pos = svc.account.current_exposure(symbol)[0]
            if abs(pos) > self.min_order_qty:
                arrs = data_arrays[symbol]
                price_idx = min(last_index, data_lengths[symbol] - 1)
                last_price = float(arrs["close"][price_idx])
                qty = abs(pos)
                side = "sell" if pos > 0 else "buy"
                exchange = self.strategy_exchange[(strat_name, symbol)]
                fee_model = self.exchange_fees.get(exchange, self.default_fee)
                trade_value = qty * last_price
                fee_cost = fee_model.calculate(trade_value, maker=False)
                prev_rpnl = getattr(svc.pos, "realized_pnl", 0.0)
                svc.on_fill(symbol, side, qty, last_price)
                new_rpnl = getattr(svc.pos, "realized_pnl", 0.0)
                slippage_pnl = 0.0
                realized_pnl = new_rpnl - prev_rpnl - fee_cost - slippage_pnl
                realized_pnl_total += realized_pnl
                svc.pos.realized_pnl = prev_rpnl + realized_pnl
                if side == "sell":
                    cash += trade_value - fee_cost
                else:
                    cash -= trade_value + fee_cost
                sync_cash()
                fill_count += 1
                if collect_fills:
                    timestamp = (
                        arrs.get("timestamp")[price_idx]
                        if "timestamp" in arrs
                        else price_idx
                    )
                    mtm_after = sum(
                        svc_s.account.current_exposure(sym_s)[0]
                        * float(
                            data_arrays[sym_s]["close"][
                                min(last_index, data_lengths[sym_s] - 1)
                            ]
                        )
                        for (strat_s, sym_s), svc_s in self.risk.items()
                    )
                    equity_after = cash + mtm_after
                    fills.append(
                        (
                            timestamp,
                            "liquidation",
                            side,
                            last_price,
                            qty,
                            strat_name,
                            symbol,
                            exchange,
                            fee_cost,
                            slippage_pnl,
                            realized_pnl,
                            realized_pnl_total,
                            equity_after,
                        )
                    )

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

        order_count = len(orders)
        cancel_count = sum(1 for o in orders if o.remaining_qty > 1e-9)
        cancel_ratio = cancel_count / order_count if order_count else 0.0

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
            "order_count": order_count,
            "cancel_count": cancel_count,
            "cancel_ratio": cancel_ratio,
            "fills": fills if collect_fills else [],
        }
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
                    "reason",
                    "side",
                    "price",
                    "qty",
                    "strategy",
                    "symbol",
                    "exchange",
                    "fee_cost",
                    "slippage_pnl",
                    "realized_pnl",
                    "realized_pnl_total",
                    "equity_after",
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
    min_fill_qty: float = MIN_FILL_QTY,
    min_order_qty: float = MIN_ORDER_QTY,
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
        min_fill_qty=min_fill_qty,
        min_order_qty=min_order_qty,
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
    min_fill_qty: float = MIN_FILL_QTY,
    min_order_qty: float = MIN_ORDER_QTY,
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
            min_fill_qty=min_fill_qty,
            min_order_qty=min_order_qty,
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
                {symbol: test_df},
                [(strategy_name, symbol)],
                latency=latency,
                window=window,
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
        raise ValueError(
            "train_size and test_size must be provided when splits is None"
        )

    start = 0
    split = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start : start + train_size].reset_index(drop=True)
        test_df = df.iloc[
            start + train_size : start + train_size + test_size
        ].reset_index(drop=True)

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

        start += test_size
        split += 1

    return results
