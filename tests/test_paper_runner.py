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
from tradingbot.core import normalize, Account
from tradingbot.utils.price import limit_price_from_close

sys.modules.pop("tradingbot.adapters.binance", None)


class DummyWS:
    def __init__(self):
        meta = types.SimpleNamespace(
            client=types.SimpleNamespace(symbols=[]),
            rules_for=lambda s: SimpleNamespace(qty_step=1e-9, price_step=0.1),
        )
        self.rest = types.SimpleNamespace(meta=meta)

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
        self.allow_short = True
        self.account = Account(float("inf"), cash=0.0)
        self.trades: dict = {}

    def mark_price(self, symbol, price):
        self.account.mark_price(symbol, price)

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


class DummyRiskWithAccount(DummyRisk):
    def __init__(self, symbol: str, *a, **k):
        super().__init__(*a, **k)
        self.account = Account(float("inf"), cash=1000.0)
        self.account.mark_price(symbol, 100.0)
        self.account.update_open_order(symbol, "buy", 0.005)

    def mark_price(self, symbol, price):
        self.account.mark_price(symbol, price)


class DummyRouter:
    def __init__(self, adapters, **kwargs):
        self.adapters = adapters


class DummyExecBroker:
    last_args = None
    maker_fee_bps = 0.0
    taker_fee_bps = 0.0

    def __init__(self, adapter):
        self.adapter = adapter

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
        return {
            "status": "filled",
            "price": price,
            "filled_qty": qty,
            "pending_qty": 0.0,
            "realized_pnl": 0.0,
        }


class DummyExecBrokerInsufficient(DummyExecBroker):
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
        DummyExecBrokerInsufficient.last_args = (
            symbol,
            side,
            price,
            qty,
            on_partial_fill,
            on_order_expiry,
        )
        return {
            "filled_qty": 0.0,
            "pending_qty": 0.0,
            "realized_pnl": 0.0,
            "reason": "insufficient_position",
        }


class DummyExecBrokerRecord(DummyExecBroker):
    called = False

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
        DummyExecBrokerRecord.called = True
        return await super().place_limit(
            symbol, side, price, qty, on_partial_fill, on_order_expiry, **_
        )


class DummyBroker:
    def __init__(
        self,
        slip_bps_per_qty: float = 0.0,
        slippage_model=None,
        **_,
    ):
        self.account = SimpleNamespace(update_cash=lambda amount: None)
        self.state = SimpleNamespace(realized_pnl=0.0, last_px={}, order_book={})

    def update_last_price(self, symbol, px):
        self.state.last_px[symbol] = px

    def equity(self, mark_prices):
        return 1000.0


class DummyServer:
    def __init__(self, config):
        self.should_exit = False

    async def serve(self):
        return


class DummyBrokerSlip(DummyBroker):
    """Record ``slip_bps_per_qty`` passed by the runner."""

    last_slip: float | None = None

    def __init__(
        self,
        slip_bps_per_qty: float = 0.0,
        slippage_model=None,
        **_,
    ):
        super().__init__(slip_bps_per_qty=slip_bps_per_qty, slippage_model=slippage_model)
        DummyBrokerSlip.last_slip = slip_bps_per_qty


class DummyRiskRecord(DummyRisk):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.on_fill_called = False

    def on_fill(self, symbol, side, qty, price=None, venue=None):
        self.on_fill_called = True


class SellStrat:
    def on_bar(self, ctx):
        return SimpleNamespace(side="sell", strength=0.37, reduce_only=False, limit_price=None)

    def on_partial_fill(self, order, res):  # pragma: no cover - simple stub
        pass

    def on_order_expiry(self, order, res):  # pragma: no cover - simple stub
        pass


class DummyRiskNoInventory(DummyRisk):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.register_called = False
        self.allow_short = False

    def register_order(self, symbol, notional):
        self.register_called = True
        return True


@pytest.mark.asyncio
async def test_run_paper(monkeypatch):
    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), DummyWS)
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": DummyStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
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

    await rp.run_paper(symbol=normalize("BTC-USDT"), strategy_name="dummy")

    sym, side, price, qty, pfill, oexp = DummyExecBroker.last_args
    assert sym == normalize("BTC-USDT")
    assert side == "buy"
    expected_price = limit_price_from_close("buy", 100.0, 0.1)
    assert price == pytest.approx(expected_price)
    assert pfill is not None and oexp is not None
    assert dummy_risk.last_strength == pytest.approx(0.37)


@pytest.mark.asyncio
async def test_run_paper_clears_existing_pending_after_fill(monkeypatch):
    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), DummyWS)
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": DummyStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    symbol = normalize("BTC-USDT")
    dummy_risk = DummyRiskWithAccount(symbol)
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

    assert dummy_risk.account.open_orders[symbol]["buy"] == pytest.approx(0.005)

    await rp.run_paper(symbol=symbol, strategy_name="dummy")

    assert dummy_risk.account.open_orders == {}
    assert getattr(dummy_risk.account, "locked_total", 0.0) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_run_paper_skips_on_fill(monkeypatch):
    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), DummyWS)
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": DummyStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    dummy_risk = DummyRiskRecord()
    monkeypatch.setattr(rp, "RiskService", lambda *a, **k: dummy_risk)
    monkeypatch.setattr(rp, "ExecutionRouter", DummyRouter)
    monkeypatch.setattr(rp, "Broker", DummyExecBrokerInsufficient)
    monkeypatch.setattr(rp, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rp.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(rp, "_CAN_PG", False)
    monkeypatch.setattr(
        rp, "settings", types.SimpleNamespace(tick_size=0.1, risk_purge_minutes=0)
    )

    await rp.run_paper(symbol=normalize("BTC-USDT"), strategy_name="dummy")

    assert dummy_risk.on_fill_called is False


@pytest.mark.asyncio
async def test_run_paper_skip_sell_no_inventory(monkeypatch):
    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), DummyWS)
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": SellStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    dummy_risk = DummyRiskNoInventory()
    monkeypatch.setattr(rp, "RiskService", lambda *a, **k: dummy_risk)
    monkeypatch.setattr(rp, "ExecutionRouter", DummyRouter)
    DummyExecBrokerRecord.called = False
    monkeypatch.setattr(rp, "Broker", DummyExecBrokerRecord)
    monkeypatch.setattr(rp, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rp.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(rp, "_CAN_PG", False)
    monkeypatch.setattr(
        rp, "settings", types.SimpleNamespace(tick_size=0.1, risk_purge_minutes=0)
    )

    await rp.run_paper(symbol=normalize("BTC-USDT"), strategy_name="dummy")

    assert dummy_risk.register_called is False
    assert DummyExecBrokerRecord.called is False


@pytest.mark.asyncio
async def test_run_paper_passes_slip(monkeypatch):
    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), DummyWS)
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": DummyStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    dummy_risk = DummyRisk()
    monkeypatch.setattr(rp, "RiskService", lambda *a, **k: dummy_risk)
    monkeypatch.setattr(rp, "ExecutionRouter", DummyRouter)
    monkeypatch.setattr(rp, "Broker", DummyExecBroker)
    monkeypatch.setattr(rp, "PaperAdapter", DummyBrokerSlip)
    monkeypatch.setattr(rp.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(rp, "_CAN_PG", False)
    monkeypatch.setattr(
        rp, "settings", types.SimpleNamespace(tick_size=0.1, risk_purge_minutes=0)
    )

    await rp.run_paper(
        symbol=normalize("BTC-USDT"), strategy_name="dummy", slip_bps_per_qty=5.0
    )

    assert DummyBrokerSlip.last_slip == 5.0
