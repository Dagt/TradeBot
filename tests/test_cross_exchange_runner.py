import pytest

from tradingbot.live.runner_cross_exchange import run_cross_exchange
from tradingbot.strategies.cross_exchange_arbitrage import CrossArbConfig
from tradingbot.execution.router import ExecutionRouter


@pytest.mark.asyncio
async def test_cross_exchange_runner_persists_and_executes(
    monkeypatch, dual_testnet
):
    spot, perp = dual_testnet
    inserted: list[dict] = []

    def fake_insert_order(engine, **kwargs):
        inserted.append(kwargs)

    class RecordingRouter(ExecutionRouter):
        def __init__(self, adapters):
            super().__init__(adapters)
            self._engine = object()

        async def best_venue(self, order):
            # route buys to spot and sells to perp for deterministic tests
            return (
                self.adapters["spot"]
                if order.side == "buy"
                else self.adapters["perp"]
            )

    monkeypatch.setattr(
        "tradingbot.live.runner_cross_exchange.ExecutionRouter",
        RecordingRouter,
    )
    monkeypatch.setattr(
        "tradingbot.execution.router.timescale.insert_order",
        fake_insert_order,
    )
    monkeypatch.setattr("tradingbot.live.runner_cross_exchange._CAN_PG", False)

    # simulate existing equity on both venues
    monkeypatch.setattr(
        "tradingbot.live.runner_cross_exchange.balances",
        {spot.name: 2.0, perp.name: 1.0},
        raising=False,
    )

    cfg = CrossArbConfig(
        symbol="BTC/USDT",
        spot=spot,
        perp=perp,
        threshold=0.001,
    )

    await run_cross_exchange(cfg)

    edge = (101.0 - 100.0) / 100.0
    equity = 2.0 * 100.0 + 1.0 * 101.0
    strength = abs(edge)
    expected_qty = min(equity * strength / 100.0, equity * strength / 101.0)

    assert spot.orders == [
        {"symbol": "BTC/USDT", "side": "buy", "qty": pytest.approx(expected_qty)}
    ]
    assert perp.orders == [
        {"symbol": "BTC/USDT", "side": "sell", "qty": pytest.approx(expected_qty)}
    ]
    assert len(inserted) == 2
    for rec in inserted:
        assert rec["qty"] == pytest.approx(expected_qty)
