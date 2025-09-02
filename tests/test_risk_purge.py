from tradingbot.risk.service import RiskService
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.core.account import Account


def test_purge_removes_inactive_symbols():
    guard = PortfolioGuard(GuardConfig(venue="X"))
    account = Account(float("inf"))
    rs = RiskService(guard, account=account, risk_pct=0.0)
    rs.update_position("X", "AAA", 1.0, entry_price=10.0)
    rs.update_position("X", "BBB", 2.0, entry_price=20.0)
    assert "AAA" in rs.trades and "BBB" in rs.trades
    rs.purge({"AAA"})
    assert set(rs.trades.keys()) == {"AAA"}
    assert "BBB" not in rs.account.positions
    assert "BBB" not in guard.st.positions
    assert "BBB" not in rs.positions_multi["X"]
