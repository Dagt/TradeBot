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

    cfg = CrossArbConfig(
        symbol="BTC/USDT",
        spot=spot,
        perp=perp,
        threshold=0.001,
        notional=100.0,
    )

    await run_cross_exchange(cfg)

    assert spot.orders == [
        {"symbol": "BTC/USDT", "side": "buy", "qty": pytest.approx(1.0)}
    ]
    assert perp.orders == [
        {"symbol": "BTC/USDT", "side": "sell", "qty": pytest.approx(1.0)}
    ]
    assert len(inserted) == 2
