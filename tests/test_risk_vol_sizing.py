import pytest

from tradingbot.risk.manager import RiskManager
from tradingbot.risk.position_sizing import delta_from_strength


def test_delta_from_strength_respects_risk_pct():
    delta = delta_from_strength(
        2.0, equity=1000.0, price=10.0, current_qty=0.0, risk_pct=0.1
    )
    assert delta == pytest.approx(10.0)


def test_risk_manager_caps_position_via_risk_pct():
    rm = RiskManager(risk_pct=0.1)
    rm.set_position(5.0)
    allowed, reason, delta = rm.check_order(
        "BTC", "buy", equity=1000.0, price=10.0, strength=1.0
    )
    assert allowed
    assert reason == ""
    assert delta == pytest.approx(5.0)
