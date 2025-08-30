import pandas as pd
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace

import types, sys

# runner_paper indirectly imports a module missing in tests; inject a stub
binance_ws_stub = types.ModuleType("tradingbot.adapters.binance_ws")
binance_ws_stub.BinanceWSAdapter = object
sys.modules.setdefault("tradingbot.adapters.binance_ws", binance_ws_stub)

from tradingbot.live import runner_paper as rp
from tradingbot.core import normalize


class DummyWS:
    async def stream_trades(self, symbol):
        yield {"ts": datetime.now(timezone.utc), "price": 100.0, "qty": 1.0}


class DummyAgg:
    def on_trade(self, ts, px, qty):
        return SimpleNamespace(c=px)

    def last_n_bars_df(self, n):
        return pd.DataFrame({"c": [1.0] * 200})


class DummyStrat:
    def on_bar(self, ctx):
        return SimpleNamespace(side="buy", strength=0.37, reduce_only=False)


class DummyRisk:
    def __init__(self, *a, **k):
        self.last_strength: float | None = None
        self.rm = types.SimpleNamespace(min_order_qty=0.0)
        self.account = types.SimpleNamespace(current_exposure=lambda symbol: (0.0, 0.0))
        self.trades: dict = {}

    def mark_price(self, symbol, price):
        pass

    def update_correlation(self, *a, **k):
        pass

    def get_trade(self, symbol):
        return self.trades.get(symbol)

    def update_trailing(self, trade, px):
        pass

    def manage_position(self, trade):
        return "hold"

    def check_order(self, symbol, side, equity, price, strength=1.0, **_):
        self.last_strength = strength
        return True, "", 1.0

    def on_fill(self, symbol, side, qty, price=None, venue=None):
        pass


class DummyRouter:
    last_order = None

    def __init__(self, adapters):
        self.adapters = {"paper": DummyBroker()}

    async def execute(self, order, algo=None, **kwargs):
        DummyRouter.last_order = order
        return {"status": "ok"}


class DummyBroker:
    def __init__(self):
        self.account = object()

    def update_last_price(self, symbol, px):
        pass

    def equity(self, mark_prices):
        return 1000.0


class DummyServer:
    def __init__(self, config):
        self.should_exit = False

    async def serve(self):
        return


@pytest.mark.asyncio
async def test_run_paper(monkeypatch):
    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": DummyStrat})
    # Use real RiskManager and PortfolioGuard to satisfy run_paper setup
    dummy_risk = DummyRisk()
    monkeypatch.setattr(rp, "RiskService", lambda *a, **k: dummy_risk)
    monkeypatch.setattr(rp, "ExecutionRouter", DummyRouter)
    monkeypatch.setattr(rp, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rp.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(rp, "_CAN_PG", False)

    await rp.run_paper(symbol=normalize("BTC-USDT"), strategy_name="dummy")

    assert DummyRouter.last_order.symbol == normalize("BTC-USDT")
    assert dummy_risk.last_strength == pytest.approx(0.37)
