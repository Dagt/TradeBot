from tradingbot.risk.manager import RiskManager


def test_stop_loss_sets_reason():
    rm = RiskManager(max_pos=1, stop_loss_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert not rm.check_limits(94)
    assert rm.enabled is False
    assert rm.last_kill_reason == "stop_loss"


def test_drawdown_sets_reason():
    rm = RiskManager(max_pos=1, max_drawdown_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert rm.check_limits(110)
    assert not rm.check_limits(104)
    assert rm.enabled is False
    assert rm.last_kill_reason == "drawdown"


def test_manual_kill_switch_records_reason():
    rm = RiskManager(max_pos=1)
    rm.kill_switch("manual")
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
