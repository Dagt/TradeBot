import pytest
from tradingbot.core import Account
from tradingbot.risk.service import RiskService
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.daily_guard import DailyGuard, GuardLimits


def _make_svc(equity=1000.0):
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=0.5, venue="X"))
    guard.refresh_usd_caps(equity)
    daily = DailyGuard(GuardLimits(), venue="X")
    account = Account(float("inf"), cash=equity)
    account.mark_price("BTC", 100.0)
    return RiskService(guard, daily, account=account)


def test_register_order_blocks_on_caps(monkeypatch):
    svc = _make_svc()
    events = []
    monkeypatch.setattr(svc, "_persist", lambda k, s, m, d: events.append(m))
    assert svc.register_order("BTC", 400.0)
    svc.account.update_open_order("BTC", 4.0)
    assert not svc.register_order("BTC", 600.0)
    assert any("per_symbol_cap_usdt" in msg for msg in events)


def test_register_order_respects_daily_guard(monkeypatch):
    svc = _make_svc()
    svc.daily._halted = True  # force halt
    events = []
    monkeypatch.setattr(svc, "_persist", lambda k, s, m, d: events.append(m))
    assert not svc.register_order("BTC", 100.0)
    assert events and events[0] == "daily_halt"
