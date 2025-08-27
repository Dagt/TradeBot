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
    strengths: list[float] = []

    def fake_insert_order(engine, **kwargs):
        inserted.append(kwargs)

    class RecordingRouter(ExecutionRouter):
        def __init__(self, adapters):
            super().__init__(adapters)
            self._engine = object()

        async def best_venue(self, order):
            return (
                self.adapters["spot"]
                if order.side == "buy"
                else self.adapters["perp"]
            )

    async def fake_check(self, symbol, side, equity, price, strength=1.0, **_):
        strengths.append(strength)
        return True, "", 1.0

    monkeypatch.setattr(
        "tradingbot.live.runner_cross_exchange.ExecutionRouter",
        RecordingRouter,
    )
    monkeypatch.setattr(
        "tradingbot.execution.router.timescale.insert_order",
        fake_insert_order,
    )
    monkeypatch.setattr(
        "tradingbot.risk.service.RiskService.check_order", fake_check, raising=False
    )
    monkeypatch.setattr(
        "tradingbot.live.runner_cross_exchange._CAN_PG", False
    )

    cfg = CrossArbConfig(
        symbol="BTC/USDT",
        spot=spot,
        perp=perp,
        threshold=0.001,
    )

    await run_cross_exchange(cfg)

    assert spot.orders == [
        {"symbol": "BTC/USDT", "side": "buy", "qty": pytest.approx(1.0)}
    ]
    assert perp.orders == [
        {"symbol": "BTC/USDT", "side": "sell", "qty": pytest.approx(1.0)}
    ]
    assert strengths == [pytest.approx(0.01), pytest.approx(0.01)]
    assert len(inserted) == 2
