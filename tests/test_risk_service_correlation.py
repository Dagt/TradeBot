from datetime import datetime, timedelta, timezone
import asyncio
import pytest
from datetime import datetime, timedelta, timezone

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.correlation_service import CorrelationService
from tradingbot.risk.service import RiskService
from tradingbot.bus import EventBus
import pandas as pd


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
    rm = RiskManager(bus=bus)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=50.0, per_symbol_cap_pct=50.0, venue="test")
    )
    guard.refresh_usd_caps(200.0)
    corr = CorrelationService()
    svc = RiskService(rm, guard, corr_service=corr, risk_pct=1.0, risk_per_trade=0.5)
    svc.account.update_cash(200.0)

    _feed_correlated_prices(corr)
    corr_df = corr._returns.corr()
    exceeded = svc.update_correlation(corr_df, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("AAA", "BBB")]
    assert events and events[0]["reason"] == "correlation"

    allowed, reason, delta = svc.check_order(
        "AAA", "buy", 100.0, corr_threshold=0.8, strength=0.5
    )
    assert allowed
    assert delta == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_risk_service_covariance_limit():
    bus = EventBus()
    events: list = []
    bus.subscribe("risk:paused", lambda e: events.append(e))
    rm = RiskManager(bus=bus)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=50.0, per_symbol_cap_pct=50.0, venue="test")
    )
    svc = RiskService(rm, guard, risk_pct=1.0)
    cov_df = pd.DataFrame(
        [[0.04, 0.039], [0.039, 0.04]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    exceeded = svc.update_covariance(cov_df, 0.8)
    await asyncio.sleep(0)
    assert exceeded == [("AAA", "BBB")]
    assert events and events[0]["reason"] == "covariance"

