import pandas as pd
import pytest

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.core import Account
from tradingbot.strategies import STRATEGIES


class DummyStrategy:
    def __init__(self, risk_service=None):
        self.risk_service = risk_service

    def on_bar(self, data):
        return None


class StubPos:
    realized_pnl = 0.0


class StubRM:
    def __init__(self):
        self.pos = StubPos()

    def check_limits(self, price):
        return True


class StubRisk:
    def __init__(self, account: Account):
        self.account = account
        self.rm = StubRM()
        self.pos = self.rm.pos

    def get_trade(self, sym):
        return {"side": "buy"}

    def update_trailing(self, trade, price):
        pass

    def manage_position(self, trade):
        return "close"

    def mark_price(self, symbol, price):
        pass

    def check_limits(self, price):
        return True

    def on_fill(self, symbol, side, qty, price):
        sign = 1 if side == "buy" else -1
        self.account.update_position(symbol, sign * qty, price)

    def complete_order(self, venue=None, *, symbol=None, side=None):
        pass


def test_pending_order_quantity(monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": [0],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    engine = EventDrivenBacktestEngine({"SYM": df}, [("dummy", "SYM")], latency=2, window=1)
    account = Account(float("inf"), cash=0.0)
    account.update_position("SYM", 1.0, price=100.0)
    engine.risk = {("dummy", "SYM"): StubRisk(account)}
    engine.run()
    assert account.open_orders["SYM"]["sell"] == pytest.approx(1.0)
