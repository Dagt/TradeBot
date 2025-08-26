import math
from datetime import datetime, timedelta, timezone

import pytest

from tradingbot.risk.correlation_service import CorrelationService
from tradingbot.risk.correlation_guard import group_correlated, global_cap
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def _feed_series(svc: CorrelationService, symbol: str, start: datetime, returns: list[float]) -> None:
    price = 100.0
    svc.update_price(symbol, price, start)
    ts = start
    for r in returns:
        ts += timedelta(minutes=1)
        price *= math.exp(r)
        svc.update_price(symbol, price, ts)


def test_correlation_service_basic():
    svc = CorrelationService(window=timedelta(hours=1))
    now = datetime.now(timezone.utc)
    rets = [0.01, -0.02, 0.03]
    _feed_series(svc, "AAA", now, rets)
    _feed_series(svc, "BBB", now, rets)
    corrs = svc.get_correlations()
    assert corrs[("AAA", "BBB")] == pytest.approx(1.0)


def test_correlation_service_window_rolls():
    svc = CorrelationService(window=timedelta(minutes=30))
    now = datetime.now(timezone.utc)
    # datos antiguos fuera de ventana
    svc.update_price("AAA", 100.0, now - timedelta(minutes=40))
    svc.update_price("BBB", 100.0, now - timedelta(minutes=40))
    svc.update_price("AAA", 101.0, now - timedelta(minutes=35))
    svc.update_price("BBB", 99.0, now - timedelta(minutes=35))
    assert svc.get_correlations() == {}
    # datos recientes dentro de ventana
    svc.update_price("AAA", 102.0, now - timedelta(minutes=10))
    svc.update_price("BBB", 102.0, now - timedelta(minutes=10))
    svc.update_price("AAA", 103.0, now - timedelta(minutes=5))
    svc.update_price("BBB", 103.0, now - timedelta(minutes=5))
    corrs = svc.get_correlations()
    assert corrs[("AAA", "BBB")] == pytest.approx(1.0)


def test_risk_service_uses_correlation_service():
    guard = PortfolioGuard(GuardConfig(per_symbol_cap_usdt=10000, total_cap_usdt=20000))
    rm = RiskManager(vol_target=0.02)
    corr = CorrelationService()
    svc = RiskService(rm, guard, corr_service=corr)
    now = datetime.now(timezone.utc)
    price_btc = 100.0
    price_eth = 200.0
    corr.update_price("BTC", price_btc, now)
    corr.update_price("ETH", price_eth, now)
    price_btc *= math.exp(0.01)
    price_eth *= math.exp(0.01)
    corr.update_price("BTC", price_btc, now + timedelta(seconds=1))
    corr.update_price("ETH", price_eth, now + timedelta(seconds=1))
    price_btc *= math.exp(0.02)
    price_eth *= math.exp(0.02)
    corr.update_price("BTC", price_btc, now + timedelta(seconds=2))
    corr.update_price("ETH", price_eth, now + timedelta(seconds=2))
    guard.st.returns["BTC"].extend([0.01, -0.02, 0.03])
    symbol_vol = guard.volatility("BTC")
    base = rm.size("buy", symbol="BTC", symbol_vol=symbol_vol)
    allowed, _, delta = svc.check_order("BTC", "buy", price=price_btc, corr_threshold=0.8)
    assert allowed
    assert delta == pytest.approx(base * 0.5)


def test_correlation_guard_groups_and_cap():
    pairs = {
        ("BTC", "ETH"): 0.9,
        ("ETH", "SOL"): 0.85,
        ("XRP", "DOGE"): 0.92,
    }
    groups = group_correlated(pairs, 0.8)
    set_groups = {frozenset(g) for g in groups}
    assert frozenset({"BTC", "ETH", "SOL"}) in set_groups
    assert frozenset({"XRP", "DOGE"}) in set_groups
    cap = global_cap(groups, 12.0)
    assert cap == pytest.approx(4.0)


def test_update_correlation_uses_guard_for_global_cap():
    rm = RiskManager()
    pairs = {
        ("BTC", "ETH"): 0.9,
        ("ETH", "SOL"): 0.85,
        ("XRP", "DOGE"): 0.7,  # below threshold
    }
    exceeded = rm.update_correlation(pairs, 0.8)
    assert set(exceeded) == {("BTC", "ETH"), ("ETH", "SOL")}
    assert rm.max_pos == pytest.approx(1.0 / 3)
