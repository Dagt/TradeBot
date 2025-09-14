import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


class FilledAdapter:
    """Adapter that immediately fills orders at a fixed price."""

    name = "paper"

    def __init__(self, price: float = 100.0):
        self.price = price
        self.state = type("S", (), {"order_book": {}, "last_px": {}})()

    async def place_order(self, **kwargs):
        return {
            "status": "filled",
            "filled_qty": kwargs["qty"],
            "price": kwargs.get("price", self.price),
            "pending_qty": 0.0,
        }


@pytest.mark.asyncio
async def test_guard_and_account_sync_after_fill():
    guard = PortfolioGuard(GuardConfig(venue="paper"))
    account = Account(float("inf"))
    risk = RiskService(guard, account=account)
    router = ExecutionRouter(FilledAdapter(), risk_service=risk)

    await router.place_order("SYM", "buy", "market", 1.0)
    assert account.positions["SYM"] == pytest.approx(1.0)
    assert guard.st.venue_positions["paper"]["SYM"] == pytest.approx(1.0)

    await router.place_order("SYM", "sell", "market", 0.4)
    assert account.positions["SYM"] == pytest.approx(0.6)
    assert guard.st.venue_positions["paper"]["SYM"] == pytest.approx(0.6)
