import pytest
from tradingbot.risk.arbitrage_service import ArbitrageRiskService, ArbGuardConfig


@pytest.mark.asyncio
async def test_risk_service_rejects_negative_edge():
    risk = ArbitrageRiskService(ArbGuardConfig(min_edge=0.0))
    notional, net = await risk.evaluate("BTC/USDT", edge=0.01, price_ref=100.0, account_equity=1000.0, fees=0.02)
    assert notional == 0.0
    assert net < 0


@pytest.mark.asyncio
async def test_risk_service_caps_position_size():
    cfg = ArbGuardConfig(max_position_usd=50.0)
    risk = ArbitrageRiskService(cfg)
    notional, _ = await risk.evaluate("BTC/USDT", edge=0.05, price_ref=100.0, account_equity=1000.0)
    assert notional == pytest.approx(50.0)
