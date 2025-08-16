
from hypothesis import given, strategies as st

def test_risk_manager_position(risk_manager):
    risk_manager.set_position(1)
    risk_manager.add_fill("buy", 2)
    assert risk_manager.pos.qty == 3
    assert risk_manager.size("buy") == 2


def test_stop_loss_triggers_disable():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(max_pos=1, stop_loss_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    # caída del 6% -> excede stop_loss_pct
    assert not rm.check_limits(94)
    assert rm.enabled is False


def test_drawdown_triggers_disable():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(max_pos=1, max_drawdown_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert rm.check_limits(110)  # peak
    # retroceso >5% desde el pico
    assert not rm.check_limits(104)
    assert rm.enabled is False


@given(signal=st.sampled_from(["buy", "sell"]), pos=st.floats(-5, 5))
def test_risk_manager_size_property(signal, pos):
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(max_pos=5)
    rm.set_position(pos)
    size = rm.size(signal)
    final = pos + size
    assert abs(final) <= rm.max_pos + 1e-12
    if signal == "buy":
        assert size >= 0
    elif signal == "sell":
        assert size <= 0
