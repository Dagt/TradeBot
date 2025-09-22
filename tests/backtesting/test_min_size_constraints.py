import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import (
    EventDrivenBacktestEngine,
    _SPOT_MIN_NOTIONAL_FALLBACK,
    _SPOT_MIN_QTY_FALLBACK,
)
from tradingbot.execution.paper import PaperAdapter
from tradingbot.risk.service import RiskService
from tradingbot.strategies import STRATEGIES


@pytest.fixture
def strategy_name(monkeypatch):
    class SingleShotStrategy:
        def __init__(self, risk_service=None):
            self._sent = False
            self.risk_service = risk_service

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0)

    monkeypatch.setitem(STRATEGIES, "min_size_strategy", SingleShotStrategy)

    def fake_calc(self, strength, price, clamp=True, **_):
        return getattr(self, "_test_qty", 0.0)

    monkeypatch.setattr(RiskService, "calc_position_size", fake_calc)
    return "min_size_strategy"


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )


def test_backtest_rejects_below_minimum(strategy_name):
    data = _sample_data()
    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [(strategy_name, "SYM", "paper_spot")],
        latency=1,
        window=1,
        exchange_configs={
            "paper_spot": {
                "market_type": "spot",
                "min_qty": 0.001,
                "step_size": 0.001,
                "min_notional": 10.0,
            }
        },
    )
    svc = next(iter(engine.risk.values()))
    svc._test_qty = 0.0005
    assert svc.min_order_qty == pytest.approx(0.001)
    assert svc.min_notional == pytest.approx(10.0)

    result = engine.run()
    assert result["fill_count"] == 0
    assert result["orders"] == []

    paper = PaperAdapter(min_notional=10.0, step_size=0.001)
    paper.state.cash = 1000.0
    paper.update_last_price("SYM", 100.0)
    paper_result = asyncio.run(
        paper.place_order("SYM", "buy", "limit", 0.0005, price=100.0)
    )
    assert paper_result["status"] == "rejected"
    assert paper_result.get("reason") == "min_notional"


def test_backtest_accepts_valid_size(strategy_name):
    data = _sample_data()
    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [(strategy_name, "SYM", "paper_spot")],
        latency=1,
        window=1,
        exchange_configs={"paper_spot": {"market_type": "spot"}},
    )
    svc = next(iter(engine.risk.values()))
    svc._test_qty = 0.2
    assert svc.min_order_qty == pytest.approx(_SPOT_MIN_QTY_FALLBACK)
    assert svc.min_notional == pytest.approx(_SPOT_MIN_NOTIONAL_FALLBACK)

    result = engine.run()
    assert result["fill_count"] >= 1
    assert result["orders"], "expected at least one order to be recorded"
    order_summary = result["orders"][0]
    assert order_summary["filled"] == pytest.approx(order_summary["qty"])

    paper = PaperAdapter(min_notional=_SPOT_MIN_NOTIONAL_FALLBACK)
    paper.state.cash = 1000.0
    paper.update_last_price("SYM", 100.0)
    paper_result = asyncio.run(
        paper.place_order(
            "SYM",
            "buy",
            "limit",
            order_summary["qty"],
            price=100.0,
        )
    )
    assert paper_result["status"] == "filled"
    assert paper_result["filled_qty"] == pytest.approx(order_summary["qty"])
