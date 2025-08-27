import pytest

from tradingbot.strategies.cross_exchange_arbitrage import (
    CrossArbConfig,
    run_cross_exchange_arbitrage,
)


class MockAdapter:
    def __init__(self, name, trades, balances):
        self.name = name
        self._trades = trades
        self.orders = []
        self._balances = balances

    async def stream_trades(self, symbol):
        for t in self._trades:
            yield t

    async def place_order(self, symbol, side, type_, qty):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        price = self._trades[0]["price"]
        return {"status": "filled", "price": price}

    async def fetch_balance(self):
        return self._balances


@pytest.mark.asyncio
async def test_rebalance_called_and_snapshots_persisted(monkeypatch):
    spot = MockAdapter("spot", [{"price": 100.0}], {"USDT": 200.0})
    perp = MockAdapter("perp", [{"price": 101.0}], {"BTC": 1.0})

    rebalance_calls = []
    async def fake_rebalance(asset, price, venues, risk, engine, threshold=0.0):
        rebalance_calls.append((asset, price, threshold))

    snapshots = []
    def fake_snapshot(engine, *, venue, symbol, position, price, notional_usd):
        snapshots.append((venue, symbol, position, price, notional_usd))

    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.rebalance_between_exchanges",
        fake_rebalance,
    )
    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.insert_portfolio_snapshot",
        fake_snapshot,
    )
    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.insert_cross_signal",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.insert_fill",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.get_engine",
        lambda: object(),
    )

    cfg = CrossArbConfig(
        symbol="BTC/USDT",
        spot=spot,
        perp=perp,
        threshold=0.001,
        strength=1.0,
        equity=100.0,
        persist_pg=True,
        rebalance_assets=("USDT",),
        rebalance_threshold=1.0,
    )

    await run_cross_exchange_arbitrage(cfg)

    assert rebalance_calls and rebalance_calls[0][0] == "USDT"
    assert rebalance_calls[0][1] == pytest.approx(1.0)
    venues = {c[0] for c in snapshots}
    assert venues == {"spot", "perp"}
    pos = {v: p for v, _, p, _, _ in snapshots}
    assert pos["spot"] == pytest.approx(1.0)
    assert pos["perp"] == pytest.approx(-1.0)
