"""Event-driven backtesting engine with order queues and slippage models."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

