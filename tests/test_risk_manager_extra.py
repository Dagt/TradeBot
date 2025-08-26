import numpy as np
import pytest

from tradingbot.risk.manager import RiskManager


def test_covariance_and_aggregation():
    rm = RiskManager()
    returns = {"A": [0.1, 0.2, 0.15], "B": [0.05, 0.07, 0.06]}
    cov = rm.covariance_matrix(returns)
    assert pytest.approx(cov[("A", "A")]) == np.var(returns["A"], ddof=1)
    assert pytest.approx(cov[("A", "B")]) == np.cov(returns["A"], returns["B"])[0, 1]

    rm.update_position("ex1", "BTC", 1.0)
    rm.update_position("ex2", "BTC", -0.3)
    agg = rm.aggregate_positions()
    assert agg["BTC"] == pytest.approx(0.7)


def test_adjust_size_and_portfolio_risk():
    rm = RiskManager()
    corr = {("A", "B"): 0.9}
    size = rm.adjust_size_for_correlation("A", 1.0, corr, 0.8)
    assert size == pytest.approx(0.5)
    size2 = rm.adjust_size_for_correlation("A", 1.0, corr, 0.95)
    assert size2 == 1.0

    cov = {("BTC", "BTC"): 0.04}
    assert rm.check_portfolio_risk({"BTC": 0.5}, cov, 1.0)
    assert not rm.check_portfolio_risk({"BTC": 0.5}, cov, 0.001)
    assert rm.enabled is False
    assert rm.last_kill_reason == "covariance_limit"


def test_de_risk_reduces_exposure():
    rm = RiskManager(vol_target=1.0)
    rm.update_pnl(100)
    assert rm.max_pos == pytest.approx(1)
    rm.update_pnl(-30)  # drawdown 30%
    assert rm.max_pos == pytest.approx(0.5)
    assert rm.vol_target == pytest.approx(0.5)
    rm.update_pnl(-25)  # drawdown 55%
    assert rm.max_pos == pytest.approx(0.25)
    assert rm.vol_target == pytest.approx(0.25)
