import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, FeeModel
from tradingbot.risk.service import RiskService
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.strategies import STRATEGIES
from tradingbot.core import Account


class BuyOnceStrategy:
    name = "buy_once"

    def __init__(self, risk_service=None):
        self.done = False
        self.risk_service = risk_service

    def on_bar(self, bar):
        if self.done:
            return SimpleNamespace(side="flat", strength=0.0)
        self.done = True
        # send a buy order well beyond available cash
        return SimpleNamespace(side="buy", strength=10.0)


class SellOnceStrategy:
    name = "sell_once"

    def __init__(self, risk_service=None):
        self.done = False
        self.risk_service = risk_service

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

    def calculate(self, cash: float, maker: bool = False) -> float:  # type: ignore[override]
        self.calls += 1
        if self.calls < 2:
            return 0.0
        return cash + 1.0  # ensure negative cash


class CheatingRiskService(RiskService):
    """Risk service that deducts double the sold quantity."""

    def on_fill(self, symbol, side, qty, price=None, venue=None, **kwargs):  # type: ignore[override]
        if side == "sell":
            qty *= 2
        super().on_fill(symbol, side, qty, price=price, venue=venue, **kwargs)


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
        risk_pct=100.0,
    )
    engine.exchange_fees["default"] = SneakyFeeModel()
    with pytest.raises(ValueError, match="negative balance"):
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
        risk_pct=100.0,
    )
    svc = engine.risk[("sell_once", "SYM")]
    cheat = CheatingRiskService(
        svc.guard,
        svc.daily,
        svc.corr,
        engine=engine,
        account=svc.account,
        risk_per_trade=svc.rm.risk_per_trade,
        atr_mult=svc.rm.atr_mult,
        risk_pct=svc.rm.risk_pct,
    )
    cheat.add_fill("buy", 1.0, price=100.0)
    cheat.update_position("default", "SYM", 1.0, entry_price=100.0)
    engine.risk[("sell_once", "SYM")] = cheat
    with pytest.raises(AssertionError, match="position went negative"):
        engine.run()


def test_spot_account_rejects_manual_cash_debit():
    account = Account(float("inf"), cash=100.0, market_type="spot")
    with pytest.raises(ValueError, match="negative balance"):
        account.update_cash(-101.0)


def test_spot_risk_service_clamps_target_volatility_size():
    guard = PortfolioGuard(GuardConfig(venue="test"))
    account = Account(float("inf"), cash=100.0, market_type="spot")
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=1.5,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    price = 10.0
    allowed, reason, delta = svc.check_order(
        "SYM",
        "buy",
        price,
        strength=1.0,
        volatility=0.5,
        target_volatility=1.5,
    )
    assert allowed is True
    assert delta * price <= account.get_available_balance() + 1e-6

