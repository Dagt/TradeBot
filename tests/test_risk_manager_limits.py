from tradingbot.risk.manager import RiskManager


def test_stop_loss_sets_reason():
    rm = RiskManager(max_pos=1, stop_loss_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert not rm.check_limits(94)
    assert rm.enabled is False
    assert rm.last_kill_reason == "stop_loss"
    assert rm.pos.qty == 0


def test_drawdown_sets_reason():
    rm = RiskManager(max_pos=1, max_drawdown_pct=0.05)
    rm.set_position(1)
    assert rm.check_limits(100)
    assert rm.check_limits(110)
    assert not rm.check_limits(104)
    assert rm.enabled is False
    assert rm.last_kill_reason == "drawdown"
    assert rm.pos.qty == 0


def test_manual_kill_switch_records_reason():
    rm = RiskManager(max_pos=1)
    rm.kill_switch("manual")
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert rm.pos.qty == 0


def test_daily_loss_limit_triggers_kill_switch():
    rm = RiskManager(max_pos=1, daily_loss_limit=50)
    rm.set_position(1)
    rm.check_limits(100)
    rm.update_pnl(-60)
    # segundo check_limits evalúa límites diarios
    assert not rm.check_limits(100)
    assert rm.enabled is False
    assert rm.last_kill_reason == "daily_loss"
    assert rm.pos.qty == 0
