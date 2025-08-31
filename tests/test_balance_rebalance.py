from unittest.mock import Mock

import asyncio
from unittest.mock import Mock

import pytest

from tradingbot.execution.balance import rebalance_between_exchanges
from tradingbot.execution.router import ExecutionRouter
from tradingbot.live.daemon import TradeBotDaemon
from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


class DummyExchange:
    def __init__(self, bal, name):
        self.bal = bal
        self.name = name
        from types import SimpleNamespace

        def _transfer(asset, amt, dest):
            self.bal -= amt
            dest.bal += amt

        self.transfer = Mock(side_effect=_transfer)

        async def fetch_balance_async():
            return {"USDT": self.bal}

        self.rest = SimpleNamespace(fetch_balance=fetch_balance_async)

    def fetch_balance(self):
        return {"USDT": {"free": self.bal}}


@pytest.mark.asyncio
async def test_rebalance_moves_funds_and_records_snapshot(monkeypatch):
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    risk = RiskService(guard, account=Account(float("inf")))
    ex_a = DummyExchange(100.0, "A")
    ex_b = DummyExchange(0.0, "B")
    engine = object()

    calls = []

    def fake_insert(engine, *, venue, symbol, position, price, notional_usd):
        calls.append((venue, position, price, notional_usd))

    monkeypatch.setattr(
        "tradingbot.execution.balance.insert_portfolio_snapshot", fake_insert
    )

    px = 2.0
    await rebalance_between_exchanges(
        "USDT",
        price=px,
        venues={"A": ex_a, "B": ex_b},
        risk=risk.rm,
        engine=engine,
        threshold=1.0,
    )

    # Transfer executed
    assert ex_a.transfer.call_count >= 1
    args = ex_a.transfer.call_args[0]
    assert args[0] == "USDT"
    assert args[1] == 50.0
    assert args[2] is ex_b

    # Risk manager updated
    assert risk.rm.positions_multi["A"]["USDT"] == pytest.approx(50.0)
    assert risk.rm.positions_multi["B"]["USDT"] == pytest.approx(50.0)

    # Snapshots recorded
    venues_recorded = {c[0] for c in calls}
    assert venues_recorded == {"A", "B"}
    for _, pos, price, notional in calls:
        assert price == px
        assert notional == pytest.approx(pos * price)


@pytest.mark.asyncio
async def test_daemon_periodic_rebalance(monkeypatch):
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    risk = RiskService(guard, account=Account(float("inf")))
    ex_a = DummyExchange(100.0, "A")
    ex_b = DummyExchange(0.0, "B")
    router = ExecutionRouter(adapters={"A": ex_a, "B": ex_b})

    calls = []
    prices_used = []

    def fake_insert(engine, *, venue, symbol, position, price, notional_usd):
        calls.append((venue, position))

    async def fake_rebalance(asset, price, **kwargs):
        prices_used.append(price)
        return await real_rebalance(asset, price=price, **kwargs)

    monkeypatch.setattr(
        "tradingbot.execution.balance.insert_portfolio_snapshot", fake_insert
    )

    real_rebalance = rebalance_between_exchanges
    monkeypatch.setattr(
        "tradingbot.live.daemon.rebalance_between_exchanges", fake_rebalance
    )

    daemon = TradeBotDaemon(
        {"A": ex_a, "B": ex_b},
        [],
        risk.rm,
        router,
        symbols=[],
        accounts={"A": ex_a, "B": ex_b},
        balance_interval=0.01,
        rebalance_assets=["USDT"],
        rebalance_threshold=1.0,
        rebalance_interval=0.01,
        rebalance_enabled=True,
    )

    daemon.last_prices["USDT"] = 2.0

    task = asyncio.create_task(daemon._rebalance_worker())
    await asyncio.sleep(0.05)
    daemon._stop.set()
    await task

    ex_a.transfer.assert_called_once()
    assert risk.rm.positions_multi["A"]["USDT"] == pytest.approx(50.0)
    assert risk.rm.positions_multi["B"]["USDT"] == pytest.approx(50.0)
    venues_recorded = {c[0] for c in calls}
    assert venues_recorded == {"A", "B"}
    assert prices_used and prices_used[0] == 2.0
