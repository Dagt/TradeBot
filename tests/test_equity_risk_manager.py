import math

from tradingbot.risk import EquityRiskManager


def test_peak_equity_and_limits():
    m = EquityRiskManager(1000, stop_loss_pct=0.1, trailing_stop_pct=0.2)
    assert m.check_limits(1000)
    assert m.check_limits(1200)
    assert math.isclose(m.peak_equity, 1200)
    assert m.check_limits(1100)  # drawdown < 20%
    assert not m.check_limits(900)  # 10% stop loss


def test_increase_and_trailing_reduction():
    m = EquityRiskManager(1000, stop_loss_pct=0.5, trailing_stop_pct=0.2)
    qty = m.increase_position("BTC", 1.0, signal=0.5, pnl_positive_threshold=10)
    assert qty > 1.0

    m.update_position_pnl("BTC", 15)
    qty2 = m.increase_position("BTC", qty, signal=0.4, pnl_positive_threshold=10)
    assert qty2 > qty

    m.update_position_pnl("BTC", 50)
    qty3 = m.reduce_position_on_trailing_stop("BTC", qty2)
    assert qty3 == qty2

    m.update_position_pnl("BTC", 35)  # 30% retrace from 50
    qty4 = m.reduce_position_on_trailing_stop("BTC", qty3)
    assert qty4 < qty3
