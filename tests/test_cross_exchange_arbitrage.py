import pytest

from tradingbot.live.runner_cross_exchange import run_cross_exchange
from tradingbot.strategies.cross_exchange_arbitrage import CrossArbConfig
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


class MockAdapter:
    def __init__(self, name, trades, order_book):
        self.name = name
        self._trades = trades
        self.orders = []
        self.state = type("S", (), {"order_book": order_book, "last_px": {}})

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


@pytest.mark.asyncio
async def test_cross_exchange_arbitrage_executes_hedged_orders():
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(101.0, 1.0)], "asks": [(102.0, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob)
    perp = MockAdapter("perp", perp_trades, perp_ob)
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001, notional=100.0)
    await run_cross_exchange(cfg)
    assert spot.orders == [{"symbol": "BTC/USDT", "side": "buy", "qty": pytest.approx(1.0)}]
    assert perp.orders == [{"symbol": "BTC/USDT", "side": "sell", "qty": pytest.approx(1.0)}]


@pytest.mark.asyncio
async def test_cross_exchange_arbitrage_no_trade_without_edge():
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 100.05, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(100.0, 1.0)], "asks": [(100.5, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob)
    perp = MockAdapter("perp", perp_trades, perp_ob)
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001, notional=100.0)
    await run_cross_exchange(cfg)
    assert spot.orders == []
    assert perp.orders == []


@pytest.mark.asyncio
async def test_cross_exchange_updates_risk_positions():
    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}]
    spot_ob = {"BTC/USDT": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    perp_ob = {"BTC/USDT": {"bids": [(101.0, 1.0)], "asks": [(102.0, 1.0)]}}
    spot = MockAdapter("spot", spot_trades, spot_ob)
    perp = MockAdapter("perp", perp_trades, perp_ob)
    cfg = CrossArbConfig(symbol="BTC/USDT", spot=spot, perp=perp, threshold=0.001, notional=100.0)
    risk = RiskService(RiskManager(), PortfolioGuard(GuardConfig(venue="test")))
    await run_cross_exchange(cfg, risk=risk)
    agg = risk.aggregate_positions()
    assert agg["BTC/USDT"] == pytest.approx(0.0)
