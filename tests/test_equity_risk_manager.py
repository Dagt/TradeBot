import pytest


from tradingbot.risk.manager import EquityRiskManager


def test_equity_risk_sizing_basic():
    rm = EquityRiskManager(risk_pct=0.1, equity_provider=lambda: 1000.0)
    qty = rm.size("buy", price=50.0)
    # 10% of 1000 = 100 -> 100 / 50 = 2
    assert qty == pytest.approx(2.0)


def test_equity_risk_returns_zero_if_not_enough():
    rm = EquityRiskManager(risk_pct=0.1, equity_provider=lambda: 20.0)
    qty = rm.size("buy", price=50.0)
    assert qty == 0.0


def test_kill_switch_disables_sizing():
    rm = EquityRiskManager(risk_pct=0.1, equity_provider=lambda: 100.0)
    rm.kill_switch("manual")
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert rm.size("buy", price=10.0) == 0.0
    rm.reset()
    assert rm.enabled is True
