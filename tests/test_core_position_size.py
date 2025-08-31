import pytest

from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


def test_core_calc_position_size_scales_with_strength():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account, risk_per_trade=0.1)
    price = 100.0
    full = rm.calc_position_size(1.0, price)
    partial = rm.calc_position_size(0.37, price)
    assert partial == pytest.approx(full * 0.37)


def test_service_calc_position_size_passes_strength():
    account = Account(float("inf"), cash=1000.0)
    guard = PortfolioGuard(GuardConfig(venue="test"))
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.1,
        atr_mult=2.0,
        risk_pct=0.01,
    )
    price = 100.0
    full = svc.calc_position_size(1.0, price)
    partial = svc.calc_position_size(0.37, price)
    assert partial == pytest.approx(full * 0.37)

    allowed, reason, delta = svc.check_order("BTC", "buy", price, strength=0.37)
    assert allowed is True
    assert reason == "caps desactivados"
    assert delta == pytest.approx(partial)


def test_calc_position_size_handles_edge_cases():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account, risk_per_trade=0.1)
    assert rm.calc_position_size(1.0, 0.0) == 0.0
    assert rm.calc_position_size(-0.5, 100.0) == 0.0


def test_initial_stop_uses_risk_pct():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account, risk_pct=0.02)
    assert rm.initial_stop(100.0, "buy") == pytest.approx(98.0)
    assert rm.initial_stop(100.0, "sell") == pytest.approx(102.0)


def test_update_trailing_advances_stop_and_stage():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account, risk_per_trade=0.1)
    trade = {
        "side": "buy",
        "entry_price": 100.0,
        "stop": 90.0,
        "atr": 5.0,
        "stage": 0,
        "qty": 1.0,
    }
    rm.update_trailing(trade, 105.0)
    assert trade["stop"] == pytest.approx(95.0)
    assert trade["stage"] == 1
    rm.update_trailing(trade, 110.0)
    assert trade["stage"] == 2
    assert trade["stop"] == pytest.approx(100.0)
    rm.update_trailing(trade, 112.0)
    assert trade["stage"] == 3
    assert trade["stop"] == pytest.approx(101.0)
    rm.update_trailing(trade, 120.0)
    assert trade["stage"] >= 4
    assert trade["stop"] == pytest.approx(110.0)


def test_manage_position_handles_stop_and_signals():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    trade = {"side": "buy", "stop": 100.0, "current_price": 99.0}
    assert rm.manage_position(trade) == "close"
    trade = {"side": "buy", "stop": 90.0, "current_price": 100.0}
    assert rm.manage_position(trade, {"side": "sell"}) == "close"
    assert rm.manage_position(trade, {"exit": True}) == "close"
    assert rm.manage_position(trade, {"side": "buy"}) == "hold"


def test_manage_position_handles_scaling():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    trade = {"side": "buy", "stop": 90.0, "current_price": 100.0, "strength": 1.0}
    assert rm.manage_position(trade, {"side": "buy", "strength": 1.5}) == "scale_in"
    assert trade["strength"] == pytest.approx(1.5)
    assert rm.manage_position(trade, {"side": "buy", "strength": 0.5}) == "scale_out"
    assert trade["strength"] == pytest.approx(0.5)
    assert rm.manage_position(trade, {"side": "buy", "strength": 0.0}) == "close"


def test_check_global_exposure_enforces_limit():
    account = Account(max_symbol_exposure=1000.0, cash=0.0)
    account.update_position("BTC", 2.0, price=100.0)
    rm = CoreRiskManager(account)
    assert rm.check_global_exposure("BTC", 700.0)
    assert not rm.check_global_exposure("BTC", 900.0)
