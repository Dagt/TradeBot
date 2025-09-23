import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, SlippageModel
from tradingbot.strategies import STRATEGIES


class FixedAdverseSlippage(SlippageModel):
    """Slippage model that enforces adverse fills for both sides."""

    def __init__(self, delta: float = 0.5) -> None:
        super().__init__(volume_impact=0.0, pct=0.0)
        self.delta = float(delta)

    def fill(
        self,
        side: str,
        qty: float,
        price: float,
        bar,
        queue_pos: float,
        partial: bool,
    ):
        frame = inspect.currentframe()
        order = frame.f_back.f_locals.get("order") if frame and frame.f_back else None
        if order is not None:
            if side == "buy":
                order.limit_price = order.place_price + self.delta
            else:
                order.limit_price = order.place_price - self.delta
        adj_price = price + (self.delta if side == "buy" else -self.delta)
        return adj_price, qty, queue_pos


def test_realized_pnl_penalizes_adverse_slippage(monkeypatch):
    class SingleFillStrategy:
        def __init__(self, risk_service=None):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=100.0)

    monkeypatch.setitem(STRATEGIES, "adverse_slip", SingleFillStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("adverse_slip", "SYM")],
        latency=1,
        window=1,
        slippage=FixedAdverseSlippage(0.5),
        verbose_fills=True,
    )

    result = engine.run()

    fills = pd.DataFrame(
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

    assert len(fills) >= 2

    first_fill = fills.iloc[0]
    second_fill = fills.iloc[1]

    assert first_fill.side == "buy"
    assert first_fill.slippage_pnl > 0

    expected_first = -(first_fill.fee_cost + first_fill.slippage_pnl)
    assert first_fill.realized_pnl == pytest.approx(expected_first)
    assert first_fill.realized_pnl_total == pytest.approx(expected_first)

    assert second_fill.side == "sell"
    assert second_fill.slippage_pnl > 0

    entry_price = first_fill.price
    exit_qty = second_fill.qty
    expected_second = (
        (second_fill.price - entry_price) * exit_qty
        - second_fill.fee_cost
        - second_fill.slippage_pnl
    )
    assert second_fill.realized_pnl == pytest.approx(expected_second)
    assert second_fill.realized_pnl_total == pytest.approx(
        first_fill.realized_pnl + second_fill.realized_pnl
    )

    total_slippage_cost = first_fill.slippage_pnl + second_fill.slippage_pnl
    assert result["slippage"] == pytest.approx(total_slippage_cost)


def test_final_liquidation_records_adverse_slippage(monkeypatch):
    class HoldLong:
        def __init__(self):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=100.0)

    monkeypatch.setitem(STRATEGIES, "hold_long_slip", HoldLong)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 80.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("hold_long_slip", "SYM")],
        latency=1,
        window=1,
        verbose_fills=True,
        fee_bps=0.0,
        risk_pct=50,
    )

    result = engine.run()

    fills = pd.DataFrame(
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

    assert len(fills) >= 2
    entry_fill = fills.iloc[0]
    liquidation = fills.iloc[-1]
    assert liquidation.reason == "liquidation"
    assert liquidation.side == "sell"

    entry_price = float(entry_fill.price)
    exit_price = float(liquidation.price)
    qty = float(liquidation.qty)
    expected_slip = (exit_price - entry_price) * qty

    assert liquidation.slippage_pnl == pytest.approx(expected_slip)
    assert liquidation.realized_pnl == pytest.approx(0.0)
    assert result["slippage"] == pytest.approx(fills["slippage_pnl"].sum())


def test_final_liquidation_records_favorable_slippage(monkeypatch):
    class HoldShort:
        def __init__(self):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="sell", strength=1.0, limit_price=100.0)

    monkeypatch.setitem(STRATEGIES, "hold_short_slip", HoldShort)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 90.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("hold_short_slip", "SYM")],
        latency=1,
        window=1,
        verbose_fills=True,
        fee_bps=0.0,
        risk_pct=50,
    )

    result = engine.run()

    fills = pd.DataFrame(
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

    assert len(fills) >= 2
    entry_fill = fills.iloc[0]
    liquidation = fills.iloc[-1]
    assert liquidation.reason == "liquidation"
    assert liquidation.side == "buy"

    entry_price = float(entry_fill.price)
    exit_price = float(liquidation.price)
    qty = float(liquidation.qty)
    expected_slip = (entry_price - exit_price) * qty

    assert liquidation.slippage_pnl == pytest.approx(expected_slip)
    assert liquidation.realized_pnl == pytest.approx(0.0)
    assert result["slippage"] == pytest.approx(fills["slippage_pnl"].sum())
