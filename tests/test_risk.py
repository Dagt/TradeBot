
import asyncio
import pytest
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


def test_size_with_volatility_event():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager(max_pos=10, vol_target=0.02)
    before = RISK_EVENTS.labels(event_type="volatility_sizing")._value.get()
    delta = rm.size_with_volatility(0.04)
    after = RISK_EVENTS.labels(event_type="volatility_sizing")._value.get()
    assert delta == 5
    assert after == before + 1


def test_update_correlation_limits_exposure():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager(max_pos=8)
    pairs = {("BTC", "ETH"): 0.9}
    before = RISK_EVENTS.labels(event_type="correlation_limit")._value.get()
    exceeded = rm.update_correlation(pairs, 0.8)
    after = RISK_EVENTS.labels(event_type="correlation_limit")._value.get()
    assert exceeded == [("BTC", "ETH")]
    assert rm.max_pos == 4
    assert after == before + 1


def test_kill_switch_disables():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.utils.metrics import RISK_EVENTS

    rm = RiskManager(max_pos=1)
    before = RISK_EVENTS.labels(event_type="kill_switch")._value.get()
    rm.kill_switch("manual")
    after = RISK_EVENTS.labels(event_type="kill_switch")._value.get()
    assert rm.enabled is False
    assert rm.last_kill_reason == "manual"
    assert after == before + 1


@pytest.mark.asyncio
async def test_daily_loss_limit_via_bus():
    from tradingbot.risk.manager import RiskManager
    from tradingbot.bus import EventBus

    bus = EventBus()
    events = []
    bus.subscribe("risk:halted", lambda e: events.append(e))
    rm = RiskManager(max_pos=1, daily_loss_limit=50, bus=bus)
    await bus.publish("pnl", {"delta": -60})
    await asyncio.sleep(0)
    assert rm.enabled is False
    assert events and events[0]["reason"] == "daily_loss"


def test_covariance_limit_triggers_kill():
    from tradingbot.risk.manager import RiskManager

    rm = RiskManager(max_pos=1)
    positions = {"BTC": 1.0, "ETH": 1.0}
    cov = {
        ("BTC", "BTC"): 0.04,
        ("ETH", "ETH"): 0.04,
        ("BTC", "ETH"): 0.039,
    }
    ok = rm.check_portfolio_risk(positions, cov, max_variance=0.1)
    assert ok is False
    assert rm.enabled is False
    assert rm.last_kill_reason == "covariance_limit"
