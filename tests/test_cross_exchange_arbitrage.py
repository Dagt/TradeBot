import pytest

from tradingbot.core import Account
from tradingbot.live.runner_cross_exchange import run_cross_exchange
from tradingbot.strategies.cross_exchange_arbitrage import (
    CrossArbConfig,
    run_cross_exchange_arbitrage,
)
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


class MockAdapter:
    def __init__(self, name, trades, order_book, balances=None):
        self.name = name
        self._trades = trades
        self.orders = []
        self.state = type("S", (), {"order_book": order_book, "last_px": {}})
        self._balances = balances or {}

    async def stream_trades(self, symbol):
        for t in self._trades:
            self.state.last_px[symbol] = t["price"]
            yield t

    async def place_order(self, symbol, side, type_, qty, price=None, post_only=False, time_in_force=None):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        px = self.state.last_px.get(symbol)
        return {"status": "filled", "price": px}

    async def cancel_order(self, order_id):  # pragma: no cover - not used
        return {"status": "canceled"}

    async def fetch_balance(self):
        return self._balances


@pytest.mark.asyncio
async def test_cross_exchange_arbitrage_executes_hedged_orders():
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(101.0, 1.0)], "asks": [(102.0, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob, {"USDT": 200.0})
    perp = MockAdapter("perp", perp_trades, perp_ob, {"BTC": 1.0})
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001)
    await run_cross_exchange_arbitrage(cfg)
    edge = (101.0 - 100.0) / 100.0
    equity = 200.0 + 1.0 * 101.0
    strength = abs(edge)
    expected_qty = min(equity * strength / 100.0, equity * strength / 101.0)
    assert spot.orders == [{"symbol": "BTC/USDT", "side": "buy", "qty": pytest.approx(expected_qty)}]
    assert perp.orders == [{"symbol": "BTC/USDT", "side": "sell", "qty": pytest.approx(expected_qty)}]


@pytest.mark.asyncio
async def test_cross_exchange_arbitrage_no_trade_without_edge():
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 100.05, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(100.0, 1.0)], "asks": [(100.5, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob, {"USDT": 200.0})
    perp = MockAdapter("perp", perp_trades, perp_ob, {"BTC": 1.0})
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001)
    await run_cross_exchange_arbitrage(cfg)
    assert spot.orders == []
    assert perp.orders == []


@pytest.mark.asyncio
async def test_cross_exchange_updates_risk_positions(monkeypatch):
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(101.0, 1.0)], "asks": [(102.0, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob, {"USDT": 200.0})
    perp = MockAdapter("perp", perp_trades, perp_ob, {"BTC": 1.0})
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001)
    guard = PortfolioGuard(GuardConfig(venue="test"))
    account = Account(float("inf"))
    risk = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.0,
    )
    monkeypatch.setattr(
        "tradingbot.live.runner_cross_exchange.balances",
        {spot.name: 2.0, perp.name: 1.0},
        raising=False,
    )
    monkeypatch.setattr("tradingbot.live.runner_cross_exchange._CAN_PG", False)
    await run_cross_exchange(cfg, risk=risk)
    agg = risk.aggregate_positions()
    assert agg["BTC/USDT"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cross_exchange_arbitrage_checks_balances():
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(101.0, 1.0)], "asks": [(102.0, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob, {"USDT": 10.0})  # insufficient
    perp = MockAdapter("perp", perp_trades, perp_ob, {"BTC": 0.0})
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001)
    await run_cross_exchange_arbitrage(cfg)
    assert spot.orders == []
    assert perp.orders == []
