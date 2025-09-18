import asyncio
import json
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
from tradingbot.utils.metrics import CANCELS
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

    def complete_order(self, venue=None, symbol=None, side=None):
        if symbol is None:
            return
        side_key = str(side).lower() if isinstance(side, str) else None
        orders = getattr(self.account, "open_orders", {})
        if not isinstance(orders, dict):
            return
        if side_key:
            try:
                orders.get(symbol, {}).pop(side_key, None)
            except Exception:
                return
            if not orders.get(symbol):
                orders.pop(symbol, None)
        else:
            orders.pop(symbol, None)


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

    async def handle_paper_event(self, ev):
        return


class DummyRouterCapture(DummyRouter):
    last_instance = None

    def __init__(self, adapters, **kwargs):
        super().__init__(adapters, **kwargs)
        DummyRouterCapture.last_instance = self
        self.on_partial_fill = None
        self.on_order_expiry = None
        self._events: list = []

    async def handle_paper_event(self, ev):
        self._events.append(ev)


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
async def test_cancel_event_after_fill_does_not_increment(monkeypatch):
    CANCELS._value.set(0)
    DummyRouterCapture.last_instance = None

    monkeypatch.setattr(rp, "BinanceWSAdapter", lambda: DummyWS())
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), DummyWS)
    monkeypatch.setattr(rp, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rp, "STRATEGIES", {"dummy": DummyStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    symbol = normalize("BTC-USDT")
    dummy_risk = DummyRiskWithAccount(symbol)
    monkeypatch.setattr(rp, "RiskService", lambda *a, **k: dummy_risk)
    monkeypatch.setattr(rp, "ExecutionRouter", DummyRouterCapture)
    monkeypatch.setattr(rp, "Broker", DummyExecBroker)
    monkeypatch.setattr(rp, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rp.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(rp, "_CAN_PG", False)
    monkeypatch.setattr(
        rp,
        "settings",
        types.SimpleNamespace(tick_size=0.1, risk_purge_minutes=0),
    )

    await rp.run_paper(symbol=symbol, strategy_name="dummy")

    router = DummyRouterCapture.last_instance
    assert router is not None and callable(router.on_order_expiry)
    _, _, _, qty, _, _ = DummyExecBroker.last_args
    order = SimpleNamespace(symbol=symbol, side="buy", pending_qty=0.0)
    cancel_event = {
        "status": "canceled",
        "filled_qty": qty,
        "pending_qty": 0.0,
        "symbol": symbol,
        "side": "buy",
    }

    router.on_order_expiry(order, dict(cancel_event))

    assert CANCELS._value.get() == 0.0
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


class AsyncFillPaperAdapter:
    maker_fee_bps = 0.0
    taker_fee_bps = 0.0

    def __init__(self, *_, **__):
        self.state = SimpleNamespace(
            realized_pnl=0.0,
            last_px={},
            order_book={},
            order_id_seq=0,
        )
        self.account = Account(float("inf"), cash=0.0)
        self.orders: dict[str, dict] = {}
        self.name = "paper"
        self.entry_px: dict[str, float] = {}
        self.default_entry = 100.0
        self.min_notional = 0.0
        self.step_size = 1e-9

    async def place_order(
        self,
        symbol,
        side,
        type_,
        qty,
        price,
        post_only=False,
        time_in_force=None,
        **_,
    ):
        order_id = f"paper-{self.state.order_id_seq}"
        self.state.order_id_seq += 1
        self.orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "price": float(price),
            "pending": float(qty),
        }
        self.state.last_px.setdefault(symbol, float(price))
        self.state.order_book.setdefault(
            symbol,
            {
                "bids": [(float(price) - 0.5, float(qty))],
                "asks": [(float(price) + 0.5, float(qty))],
            },
        )
        return {
            "status": "new",
            "order_id": order_id,
            "pending_qty": float(qty),
            "filled_qty": 0.0,
            "price": float(price),
        }

    async def cancel_order(self, order_id, symbol):  # pragma: no cover - defensive
        self.orders.pop(str(order_id), None)
        return {"status": "canceled", "order_id": order_id, "symbol": symbol, "pending_qty": 0.0}

    def update_last_price(self, symbol, px):
        self.state.last_px[symbol] = float(px)
        self.account.mark_price(symbol, float(px))
        fills: list[dict] = []
        for oid, order in list(self.orders.items()):
            side = order["side"]
            price = order["price"]
            pending = order["pending"]
            if pending <= 0:
                continue
            crosses = (side == "sell" and px >= price) or (side == "buy" and px <= price)
            if not crosses:
                continue
            fill_qty = pending
            order["pending"] = 0.0
            entry = self.entry_px.get(symbol, self.default_entry)
            if side == "sell":
                pnl_delta = (float(px) - float(entry)) * fill_qty
                self.account.update_position(symbol, -fill_qty, price=float(px))
            else:
                pnl_delta = (float(entry) - float(px)) * fill_qty
                self.account.update_position(symbol, fill_qty, price=float(px))
            self.state.realized_pnl += pnl_delta
            fills.append(
                {
                    "status": "filled",
                    "order_id": oid,
                    "symbol": symbol,
                    "side": side,
                    "filled_qty": fill_qty,
                    "pending_qty": 0.0,
                    "price": float(px),
                    "realized_pnl": self.state.realized_pnl,
                }
            )
            self.orders.pop(oid, None)
        return fills


class AsyncDummyWS:
    def __init__(self, trades):
        meta = types.SimpleNamespace(
            client=types.SimpleNamespace(symbols=[]),
            rules_for=lambda s: SimpleNamespace(qty_step=1e-9, price_step=0.1, min_notional=0.0),
        )
        self.rest = types.SimpleNamespace(meta=meta)
        self._trades = trades

    async def stream_trades(self, symbol):
        for trade in self._trades:
            yield trade


class CloseRisk(DummyRisk):
    def __init__(self, symbol: str):
        super().__init__()
        self.account = Account(float("inf"), cash=0.0)
        self.account.update_position(symbol, 1.0, price=100.0)
        self.account.mark_price(symbol, 100.0)
        self._symbol = symbol
        self._closed = False

    def get_trade(self, symbol):
        if symbol != self._symbol or self._closed:
            return None
        return {"side": "buy"}

    def manage_position(self, trade):
        if self._closed:
            return "hold"
        self._closed = True
        return "close"

    def complete_order(self, *_, **__):
        pass


class NoopStrat:
    warmup_bars = 0

    def on_bar(self, _ctx):
        return None

    def on_partial_fill(self, order, res):  # pragma: no cover - simple stub
        return None

    def on_order_expiry(self, order, res):  # pragma: no cover - simple stub
        return None


class TestAgg:
    def __init__(self, *_, **__):
        self.completed = [SimpleNamespace(c=100.0)]

    def on_trade(self, ts, px, qty):
        bar = SimpleNamespace(c=px)
        self.completed.append(bar)
        return bar

    def last_n_bars_df(self, n):
        return pd.DataFrame({"c": [100.0] * n})


class DummyCorr:
    def get_correlations(self):
        return {}


@pytest.mark.asyncio
async def test_run_paper_async_fill_updates_trade_metrics(monkeypatch):
    symbol = normalize("BTC-USDT")
    trades = [
        {"ts": datetime.now(timezone.utc), "price": 100.0, "qty": 1.0},
        {"ts": datetime.now(timezone.utc), "price": 105.0, "qty": 1.0},
    ]

    def ws_factory():
        return AsyncDummyWS(trades)

    def adapter_factory(**_):
        adapter = AsyncFillPaperAdapter()
        adapter.entry_px[symbol] = 100.0
        adapter.state.order_book[symbol] = {
            "bids": [(99.5, 1.0)],
            "asks": [(100.5, 1.0)],
        }
        return adapter

    def risk_factory(*_, **__):
        return CloseRisk(symbol)

    async def dummy_start_metrics(port):
        return SimpleNamespace(servers=[]), asyncio.create_task(asyncio.sleep(0))

    metrics: list[dict] = []

    original_info = rp.log.info

    def capture(msg, *args, **kwargs):
        if str(msg).startswith("METRICS") and args:
            try:
                metrics.append(json.loads(args[0]))
            except (json.JSONDecodeError, TypeError):
                pass
        if callable(original_info):
            return original_info(msg, *args, **kwargs)

    monkeypatch.setattr(rp, "BinanceWSAdapter", ws_factory)
    monkeypatch.setitem(rp.WS_ADAPTERS, ("binance", "spot"), lambda: AsyncDummyWS(trades))
    monkeypatch.setattr(rp, "BarAggregator", TestAgg)
    monkeypatch.setattr(rp, "PaperAdapter", adapter_factory)
    monkeypatch.setattr(rp, "RiskService", risk_factory)
    monkeypatch.setattr(rp, "PortfolioGuard", lambda *a, **k: SimpleNamespace(cfg=SimpleNamespace(venue="paper"), refresh_usd_caps=lambda cash: None))
    monkeypatch.setattr(rp, "CorrelationService", lambda *a, **k: DummyCorr())
    monkeypatch.setattr(rp, "STRATEGIES", {"noop": NoopStrat})
    monkeypatch.setattr(rp, "REST_ADAPTERS", {})
    monkeypatch.setattr(rp, "_CAN_PG", False)
    monkeypatch.setattr(rp, "_start_metrics", dummy_start_metrics)
    monkeypatch.setattr(rp.ccxt.Exchange, "parse_timeframe", staticmethod(lambda _: 0))
    monkeypatch.setattr(rp.log, "info", capture)
    monkeypatch.setattr(
        rp,
        "settings",
        types.SimpleNamespace(tick_size=0.1, risk_purge_minutes=0),
    )

    await rp.run_paper(symbol=symbol, strategy_name="noop", metrics_port=0)

    trade_logs = [entry for entry in metrics if entry.get("event") == "trade"]
    assert trade_logs, "expected trade metrics entry"
    assert trade_logs[-1]["trades_closed"] == 1
    assert trade_logs[-1]["trade_pnl"] == pytest.approx(5.0)

    total_logs = [entry for entry in metrics if list(entry.keys()) == ["pnl"]]
    assert total_logs, "expected cumulative pnl log"
    assert total_logs[-1]["pnl"] == pytest.approx(5.0)
