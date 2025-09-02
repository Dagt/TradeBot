import threading
import pytest

from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def _make_rs() -> RiskService:
    account = Account(float("inf"), cash=1_000_000.0)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    return RiskService(guard, account=account, risk_pct=0.0, risk_per_trade=1.0)


def test_concurrent_on_fill_consistency():
    rs = _make_rs()

    def worker():
        rs.on_fill("BTC", "buy", 1.0, price=100.0, venue="X")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert rs.account.positions["BTC"] == pytest.approx(10.0)
    assert rs.positions_multi["X"]["BTC"] == pytest.approx(10.0)
    trade = rs.get_trade("BTC")
    assert trade["qty"] == pytest.approx(10.0)
    assert trade["side"] == "buy"


def test_concurrent_update_position_consistency():
    rs = _make_rs()

    def worker(qty: float):
        rs.update_position("X", "BTC", qty, entry_price=100.0)

    threads = [threading.Thread(target=worker, args=(i + 1.0,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    qty = rs.account.positions["BTC"]
    assert rs.positions_multi["X"]["BTC"] == pytest.approx(qty)
    trade = rs.get_trade("BTC")
    assert trade["qty"] == pytest.approx(abs(qty))
