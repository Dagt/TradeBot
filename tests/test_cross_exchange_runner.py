import pytest

from tradingbot.strategies.cross_exchange_arbitrage import (
    run_cross_exchange_arbitrage,
    CrossArbConfig,
)


class DummyAdapter:
    def __init__(self, name, trades):
        self.name = name
        self._trades = trades
        self.orders = []
        self.state = type("S", (), {"last_px": {}})

    async def stream_trades(self, symbol):
        for t in self._trades:
            self.state.last_px[symbol] = t["price"]
            yield t

    async def place_order(self, symbol, side, type_, qty, price=None, post_only=False, time_in_force=None):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        return {"status": "filled", "price": self.state.last_px[symbol]}

    async def cancel_order(self, order_id):
        return {"status": "canceled"}


@pytest.mark.asyncio
async def test_cross_exchange_runner_persists_and_executes(monkeypatch):
    spot = DummyAdapter("spot", [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}])
    perp = DummyAdapter("perp", [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}])

    signals = []
    fills = []

    def fake_get_engine():
        return object()

    def fake_insert_cross_signal(engine, **kwargs):
        signals.append(kwargs)

    def fake_insert_fill(engine, **kwargs):
        fills.append(kwargs)

    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.get_engine",
        fake_get_engine,
    )
    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.insert_cross_signal",
        fake_insert_cross_signal,
    )
    monkeypatch.setattr(
        "tradingbot.strategies.cross_exchange_arbitrage.insert_fill",
        fake_insert_fill,
    )

    cfg = CrossArbConfig(
        symbol="BTC/USDT",
        spot=spot,
        perp=perp,
        threshold=0.001,
        notional=100.0,
        persist_pg=True,
    )

    await run_cross_exchange_arbitrage(cfg)

    assert spot.orders == [{"symbol": "BTC/USDT", "side": "buy", "qty": pytest.approx(1.0)}]
    assert perp.orders == [{"symbol": "BTC/USDT", "side": "sell", "qty": pytest.approx(1.0)}]
    assert len(signals) == 1
    assert len(fills) == 2
