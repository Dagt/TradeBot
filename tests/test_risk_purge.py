from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.core.account import Account


def test_risk_service_purge_removes_inactive_symbols():
    guard = PortfolioGuard(GuardConfig(venue="test"))
    account = Account(float("inf"))
    rs = RiskService(guard, account=account)

    rs.update_position("v", "AAA", 1.0)
    rs.update_position("v", "BBB", 2.0)

    rs.purge(["AAA"])

    assert "BBB" not in rs.trades
    assert "BBB" not in rs.positions_multi.get("v", {})
    assert "BBB" not in rs.account.positions
    assert "BBB" not in guard.st.positions
