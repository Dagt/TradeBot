from unittest.mock import Mock

import asyncio
from unittest.mock import Mock

import pytest

from tradingbot.execution.balance import rebalance_between_exchanges
from tradingbot.risk.manager import RiskManager
from tradingbot.execution.router import ExecutionRouter
from tradingbot.live.daemon import TradeBotDaemon


class DummyExchange:
    def __init__(self, bal):
        self.bal = bal
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
    risk = RiskManager()
    ex_a = DummyExchange(100.0)
    ex_b = DummyExchange(0.0)
    engine = object()

    calls = []

    def fake_insert(engine, *, venue, symbol, position, price, notional_usd):
        calls.append((venue, position, price, notional_usd))

    monkeypatch.setattr(
        "tradingbot.execution.balance.insert_portfolio_snapshot", fake_insert
    )

    await rebalance_between_exchanges(
        "USDT",
        price=1.0,
        venues={"A": ex_a, "B": ex_b},
        risk=risk,
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
    assert risk.positions_multi["A"]["USDT"] == pytest.approx(50.0)
    assert risk.positions_multi["B"]["USDT"] == pytest.approx(50.0)

    # Snapshots recorded
    venues_recorded = {c[0] for c in calls}
    assert venues_recorded == {"A", "B"}
    for _, pos, price, notional in calls:
        assert price == 1.0
        assert notional == pytest.approx(pos * price)


@pytest.mark.asyncio
async def test_daemon_periodic_rebalance(monkeypatch):
    risk = RiskManager()
    ex_a = DummyExchange(100.0)
    ex_b = DummyExchange(0.0)
    router = ExecutionRouter(adapters={"A": ex_a, "B": ex_b})

    calls = []

    def fake_insert(engine, *, venue, symbol, position, price, notional_usd):
        calls.append((venue, position))

    monkeypatch.setattr(
        "tradingbot.execution.balance.insert_portfolio_snapshot", fake_insert
    )

    daemon = TradeBotDaemon(
        {"A": ex_a, "B": ex_b},
        [],
        risk,
        router,
        symbols=[],
        accounts={"A": ex_a, "B": ex_b},
        balance_interval=0.01,
        rebalance_assets=["USDT"],
        rebalance_threshold=1.0,
    )

    task = asyncio.create_task(daemon._balance_worker())
    await asyncio.sleep(0.05)
    daemon._stop.set()
    await task

    ex_a.transfer.assert_called_once()
    assert risk.positions_multi["A"]["USDT"] == pytest.approx(50.0)
    assert risk.positions_multi["B"]["USDT"] == pytest.approx(50.0)
    venues_recorded = {c[0] for c in calls}
    assert venues_recorded == {"A", "B"}
