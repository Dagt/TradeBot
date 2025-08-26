from datetime import datetime, timedelta, timezone
import asyncio
import pytest
from datetime import datetime, timedelta, timezone

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.correlation_service import CorrelationService
from tradingbot.risk.service import RiskService
from tradingbot.bus import EventBus


def _feed_correlated_prices(cs: CorrelationService) -> None:
    start = datetime.now(timezone.utc)
    prices_a = [100, 102, 101]
    prices_b = [50, 52, 51]
    for i, (pa, pb) in enumerate(zip(prices_a, prices_b)):
        ts = start + timedelta(seconds=i)
        cs.update_price("AAA", pa, ts)
        cs.update_price("BBB", pb, ts)


@pytest.mark.asyncio
async def test_risk_service_correlation_limits_and_sizing():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:paused", lambda e: events.append(e))
    rm = RiskManager(equity_pct=1.0, bus=bus)
    guard = PortfolioGuard(
        GuardConfig(total_cap_usdt=1000.0, per_symbol_cap_usdt=500.0, venue="test")
    )
    guard.equity = 200.0
    corr = CorrelationService()
    svc = RiskService(rm, guard, corr_service=corr)

    _feed_correlated_prices(corr)
    exceeded = svc.update_correlation(0.8)
    await asyncio.sleep(0)
    assert exceeded == [("AAA", "BBB")]
    assert rm.equity_pct == pytest.approx(0.5)
    assert events and events[0]["reason"] == "correlation"

    allowed, _, delta = svc.check_order(
        "AAA", "buy", 100.0, corr_threshold=0.8
    )
    assert allowed
    assert delta == pytest.approx(0.5)

