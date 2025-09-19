import asyncio
import contextlib
import json
import logging
import pandas as pd
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace

import types, sys

# runner_paper indirectly imports a module missing in tests; inject a stub
binance_stub = types.ModuleType("tradingbot.adapters.binance")
binance_stub.BinanceWSAdapter = object
sys.modules.setdefault("tradingbot.adapters.binance", binance_stub)

from tradingbot.live import runner_paper as rp
from tradingbot.core import normalize
from tradingbot.live.paper_utils import process_paper_fills
from tradingbot.execution.paper import PaperAdapter
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService

sys.modules.pop("tradingbot.adapters.binance", None)


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
        return SimpleNamespace(side="buy", strength=0.37, reduce_only=False, limit_price=None)

    def on_partial_fill(self, order, res):  # pragma: no cover - simple stub
        pass

    def on_order_expiry(self, order, res):  # pragma: no cover - simple stub
        pass


class DummyRisk:
    def __init__(self, *a, **k):
        self.last_strength: float | None = None
        self.min_order_qty = 0.0
        self.rm = types.SimpleNamespace(allow_short=True)
        self.account = types.SimpleNamespace(
            current_exposure=lambda symbol: (0.0, 0.0),
            update_open_order=lambda symbol, side, qty: None,
            open_orders={},
            cash=0.0,
        )
        self.account.get_available_balance = lambda: self.account.cash
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

    def check_order(self, symbol, side, price, strength=1.0, **_):
        self.last_strength = strength
        return True, "", 1.0

    def register_order(self, symbol, notional):
        return True

    def on_fill(self, symbol, side, qty, price=None, venue=None):
        pass

    def daily_mark(self, broker, symbol, price, delta_rpnl):
        return False, ""

    def purge(self, symbols):
        pass


class DummyRouter:
    def __init__(self, adapters, **kwargs):
        self.adapters = adapters


class DummyExecBroker:
    last_args = None

    def __init__(self, adapter):
        self.adapter = adapter
        self.maker_fee_bps = 0.0

    async def place_limit(
        self,
        symbol,
        side,
        price,
        qty,
        on_partial_fill=None,
        on_order_expiry=None,
        **_,
    ):
        DummyExecBroker.last_args = (symbol, side, price, qty, on_partial_fill, on_order_expiry)
        return {"filled_qty": qty, "pending_qty": 0.0, "realized_pnl": 0.0}


class DummyBroker:
    def __init__(self):
        self.account = object()
        self.state = SimpleNamespace(realized_pnl=0.0, last_px={}, order_book={})
        self.taker_fee_bps = 0.0

    def update_last_price(self, symbol, px):
        self.state.last_px[symbol] = px

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
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), lambda: DummyWS())
    # Use real RiskManager and PortfolioGuard to satisfy run_paper setup
    dummy_risk = DummyRisk()
    monkeypatch.setattr(rp, "RiskService", lambda *a, **k: dummy_risk)
    monkeypatch.setattr(rp, "ExecutionRouter", DummyRouter)
    monkeypatch.setattr(rp, "Broker", DummyExecBroker)
    monkeypatch.setattr(rp, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rp.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(rp, "_CAN_PG", False)
    monkeypatch.setattr(
        rp,
        "settings",
        types.SimpleNamespace(tick_size=0.1, risk_purge_minutes=0),
    )

    DummyExecBroker.last_args = None

    task = asyncio.create_task(
        rp.run_paper(symbol=normalize("BTC-USDT"), strategy_name="dummy")
    )
    try:
        async def _wait_for_order():
            while DummyExecBroker.last_args is None:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait_for_order(), timeout=5.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    sym, side, price, qty, pfill, oexp = DummyExecBroker.last_args
    assert sym == normalize("BTC-USDT")
    assert side == "buy"
    assert price == pytest.approx(99.9)
    assert pfill is not None and oexp is not None
    assert dummy_risk.last_strength == pytest.approx(0.37)


@pytest.mark.asyncio
async def test_process_paper_fills_clears_open_orders(caplog):
    symbol = "BTC/USDT"
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.account.cash = 1000.0
    risk = RiskService(PortfolioGuard(GuardConfig(venue="paper")), account=adapter.account, risk_pct=0.0)

    adapter.update_last_price(symbol, 100.0)
    resp = await adapter.place_order(symbol, "buy", "limit", 1.0, price=90.0)
    pending_qty = float(resp.get("pending_qty", 0.0))
    assert pending_qty > 0.0

    risk.account.mark_price(symbol, 100.0)
    risk.account.update_open_order(symbol, "buy", pending_qty)
    locked_before = risk.account.cash - risk.account.get_available_balance()
    assert locked_before > 0.0

    fills = adapter.update_last_price(symbol, 90.0)
    assert fills
    logger = logging.getLogger("test.paper_fills")
    with caplog.at_level(logging.INFO):
        process_paper_fills(fills, risk, symbol, logger=logger)

    assert risk.account.open_orders == {}
    assert risk.account.get_available_balance() == pytest.approx(risk.account.cash)

    locked_records = [
        json.loads(record.message.split("METRICS ", 1)[1])
        for record in caplog.records
        if "METRICS" in record.message and "locked_cash" in record.message
    ]
    assert locked_records, "locked cash metric was not emitted"
    assert locked_records[-1]["value"] == pytest.approx(0.0)
