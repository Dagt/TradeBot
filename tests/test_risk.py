
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
