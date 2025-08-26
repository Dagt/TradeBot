import pytest

from tradingbot.execution.paper import PaperAdapter
from tradingbot.risk.manager import EquityRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


def test_portfolio_guard_blocks_insufficient_cash():
    broker = PaperAdapter()
    broker.state.cash = 100.0
    pg = PortfolioGuard(GuardConfig(total_cap_usdt=1000.0, per_symbol_cap_usdt=1000.0, venue="paper"), provider=broker)
    pg.mark_price("BTC/USDT", 50.0)
    action, reason, _ = pg.soft_cap_decision("BTC/USDT", "buy", 3.0, 50.0)
    assert action == "block"
    assert reason == "insufficient_cash"


def test_equity_risk_manager_blocks_on_cash():
    broker = PaperAdapter()
    broker.state.cash = 100.0
    rm = EquityRiskManager(max_pos=10, provider=broker)
    pg = PortfolioGuard(GuardConfig(total_cap_usdt=1000.0, per_symbol_cap_usdt=1000.0, venue="paper"), provider=broker)
    risk = RiskService(rm, pg)
    pg.mark_price("BTC/USDT", 50.0)
    allowed, reason, _ = risk.check_order("BTC/USDT", "buy", 50.0)
    assert not allowed
    assert reason == "insufficient_cash"
