import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, FeeModel
from tradingbot.risk.service import RiskService
from tradingbot.strategies import STRATEGIES


class BuyOnceStrategy:
    name = "buy_once"

    def __init__(self):
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return SimpleNamespace(side="flat", strength=0.0)
        self.done = True
        # send a buy order well beyond available cash
        return SimpleNamespace(side="buy", strength=10.0)


class SellOnceStrategy:
    name = "sell_once"

    def __init__(self):
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return SimpleNamespace(side="flat", strength=0.0)
        self.done = True
        # send a sell order well beyond held position
        return SimpleNamespace(side="sell", strength=10.0)


class SneakyFeeModel(FeeModel):
    """Fee model that returns zero initially then a very large fee."""

    def __init__(self):
        super().__init__(0.0)
        self.calls = 0

    def calculate(self, cash: float) -> float:  # type: ignore[override]
        self.calls += 1
        if self.calls < 2:
            return 0.0
        return cash + 1.0  # ensure negative cash


class CheatingRiskService(RiskService):
    """Risk service that deducts double the sold quantity."""

    def on_fill(self, symbol, side, qty, price=None, venue=None):  # type: ignore[override]
        if side == "sell":
            qty *= 2
        super().on_fill(symbol, side, qty, price, venue)


def _make_data():
    rng = pd.date_range("2021-01-01", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000,
        }
    )
    return {"SYM": df}


def test_buy_order_exceeding_cash_triggers_assert(monkeypatch):
    monkeypatch.setitem(STRATEGIES, "buy_once", BuyOnceStrategy)
    data = _make_data()
    strategies = [("buy_once", "SYM")]
    engine = EventDrivenBacktestEngine(
        data,
        strategies,
        latency=1,
        window=1,
        exchange_configs={"default": {"market_type": "spot"}},
        initial_equity=100.0,
        risk_pct=1.0,
    )
    engine.exchange_fees["default"] = SneakyFeeModel()
    with pytest.raises(AssertionError, match="cash became negative"):
        engine.run()


def test_sell_order_exceeding_position_triggers_assert(monkeypatch):
    monkeypatch.setitem(STRATEGIES, "sell_once", SellOnceStrategy)
    data = _make_data()
    strategies = [("sell_once", "SYM")]
    engine = EventDrivenBacktestEngine(
        data,
        strategies,
        latency=1,
        window=1,
        exchange_configs={"default": {"market_type": "spot"}},
        initial_equity=1000.0,
        risk_pct=1.0,
    )
    svc = engine.risk[("sell_once", "SYM")]
    svc.rm.set_position(1.0)
    engine.risk[("sell_once", "SYM")] = CheatingRiskService(
        svc.guard,
        svc.daily,
        svc.corr,
        engine=engine,
        risk_pct=svc.core.risk_pct,
    )
    with pytest.raises(AssertionError, match="position went negative"):
        engine.run()

