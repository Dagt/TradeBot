import numpy as np
import numpy as np
import pytest

from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def _make_rs() -> RiskService:
    account = Account(float("inf"), cash=0.0)
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="test")
    )
    return RiskService(guard, account=account, risk_pct=0.0, risk_per_trade=1.0)


def test_covariance_and_aggregation():
    rs = _make_rs()
    returns = {"A": [0.1, 0.2, 0.15], "B": [0.05, 0.07, 0.06]}
    cov = rs.covariance_matrix(returns)
    assert pytest.approx(cov[("A", "A")]) == np.var(returns["A"], ddof=1)
    assert pytest.approx(cov[("A", "B")]) == np.cov(returns["A"], returns["B"])[0, 1]

    rs.update_position("ex1", "BTC", 1.0)
    rs.update_position("ex2", "BTC", -0.3)
    agg = rs.aggregate_positions()
    assert agg["BTC"] == pytest.approx(0.7)


def test_adjust_size_and_portfolio_risk():
    rs = _make_rs()
    corr = {("A", "B"): 0.9}
    size = rs.adjust_size_for_correlation("A", 2.0, corr, 0.8)
    assert size == pytest.approx(1.0)
    size2 = rs.adjust_size_for_correlation("A", 2.0, corr, 0.95)
    assert size2 == 2.0

    cov = {("BTC", "BTC"): 0.04}
    assert rs.check_portfolio_risk({"BTC": 0.5}, cov, 1.0)
    assert not rs.check_portfolio_risk({"BTC": 0.5}, cov, 0.001)
    assert rs.enabled is False
    assert rs.last_kill_reason == "covariance_limit"

