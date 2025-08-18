import pandas as pd
import pandas as pd
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
import importlib, sys, types, os

# runner_testnet indirectly imports a module missing in tests; inject a stub
binance_ws_stub = types.ModuleType("tradingbot.adapters.binance_ws")
binance_ws_stub.BinanceWSAdapter = object
sys.modules.setdefault("tradingbot.adapters.binance_ws", binance_ws_stub)

from tradingbot.live import runner_testnet as rt
from tradingbot.live import runner_real as rr


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
        return SimpleNamespace(side="buy", strength=1.0)


class DummyRisk:
    def size(self, side, strength):
        return 1.0

    def check_limits(self, price):
        return True

    def add_fill(self, side, qty):
        pass


class DummyPG:
    def mark_price(self, symbol, px):
        pass

    def soft_cap_decision(self, *a):
        return ("allow", "", None)


class DummyDG:
    def on_mark(self, *a, **k):
        pass

    def check_halt(self, broker=None):
        return (False, "")


class DummyBroker:
    def __init__(self, fee_bps=0):
        pass

    def update_last_price(self, symbol, px):
        pass

    def equity(self, mark_prices):
        return 1000.0


class DummyExec:
    last_instance = None

    def __init__(self, leverage=None, testnet=False):
        self.leverage = leverage
        self.testnet = testnet
        self.orders = []
        DummyExec.last_instance = self

    async def place_order(self, symbol, side, type_, qty, mark_price=None):
        self.orders.append((symbol, side, type_, qty))
        return {"status": "ok"}


@pytest.mark.asyncio
async def test_bybit_futures_order(monkeypatch):
    monkeypatch.setattr(rt, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rt, "BreakoutATR", lambda: DummyStrat())
    monkeypatch.setattr(rt, "RiskManager", lambda max_pos: DummyRisk())
    monkeypatch.setattr(rt, "PortfolioGuard", lambda config: DummyPG())
    monkeypatch.setattr(rt, "DailyGuard", lambda limits, venue: DummyDG())
    monkeypatch.setattr(rt, "PaperAdapter", DummyBroker)

    monkeypatch.setitem(
        rt.ADAPTERS,
        ("bybit", "futures"),
        (lambda: DummyWS(), DummyExec, "bybit_futures_testnet"),
    )

    cfg = rt._SymbolConfig(symbol="BTC/USDT", trade_qty=1.0)
    await rt._run_symbol(
        "bybit",
        "futures",
        cfg,
        leverage=5,
        dry_run=False,
        total_cap_usdt=1000.0,
        per_symbol_cap_usdt=500.0,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_usdt=100.0,
        daily_max_drawdown_pct=0.05,
        max_consecutive_losses=3,
    )

    inst = DummyExec.last_instance
    assert inst.leverage == 5
    assert inst.testnet is True
    assert inst.orders[0][:4] == ("BTC/USDT", "buy", "market", 1.0)


class DummyExec2(DummyExec):
    pass


@pytest.mark.asyncio
async def test_okx_futures_order(monkeypatch):
    monkeypatch.setattr(rt, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rt, "BreakoutATR", lambda: DummyStrat())
    monkeypatch.setattr(rt, "RiskManager", lambda max_pos: DummyRisk())
    monkeypatch.setattr(rt, "PortfolioGuard", lambda config: DummyPG())
    monkeypatch.setattr(rt, "DailyGuard", lambda limits, venue: DummyDG())
    monkeypatch.setattr(rt, "PaperAdapter", DummyBroker)

    monkeypatch.setitem(
        rt.ADAPTERS,
        ("okx", "futures"),
        (lambda: DummyWS(), DummyExec2, "okx_futures_testnet"),
    )

    cfg = rt._SymbolConfig(symbol="BTC/USDT", trade_qty=1.0)
    await rt._run_symbol(
        "okx",
        "futures",
        cfg,
        leverage=7,
        dry_run=False,
        total_cap_usdt=1000.0,
        per_symbol_cap_usdt=500.0,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_usdt=100.0,
        daily_max_drawdown_pct=0.05,
        max_consecutive_losses=3,
    )

    inst = DummyExec2.last_instance
    assert inst.leverage == 7
    assert inst.testnet is True
    assert inst.orders[0][:4] == ("BTC/USDT", "buy", "market", 1.0)


class RealWS:
    async def stream_trades(self, symbol):
        yield {"price": 100.0}


class RealPaper:
    last_instance = None

    def __init__(self):
        self.orders = []
        RealPaper.last_instance = self

    async def place_order(self, symbol, side, type_, qty):
        self.orders.append((symbol, side, type_, qty))


class RealExec:
    last_instance = None

    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.orders = []
        RealExec.last_instance = self

    async def place_order(self, symbol, side, type_, qty):
        self.orders.append((symbol, side, type_, qty))


@pytest.mark.asyncio
async def test_run_real_requires_confirmation():
    with pytest.raises(RuntimeError):
        await rr.run_real(i_know_what_im_doing=False)


@pytest.mark.asyncio
async def test_run_real_dry_run(monkeypatch):
    monkeypatch.setattr(rr, "BinanceSpotWSAdapter", lambda: RealWS())
    monkeypatch.setattr(rr, "PaperAdapter", RealPaper)

    def _fail_exec(*a, **k):
        raise AssertionError("real adapter should not be used in dry run")

    monkeypatch.setattr(rr, "BinanceSpotAdapter", _fail_exec)

    calls = {"dotenv": 0}

    def fake_load():
        calls["dotenv"] += 1
        os.environ["BINANCE_API_KEY"] = "k"
        os.environ["BINANCE_API_SECRET"] = "s"

    monkeypatch.setattr(rr, "load_dotenv", fake_load)

    await rr.run_real(dry_run=True, i_know_what_im_doing=True)

    assert calls["dotenv"] == 1
    assert RealPaper.last_instance.orders[0][:4] == ("BTC/USDT", "buy", "market", 0.001)


@pytest.mark.asyncio
async def test_run_real_real_mode(monkeypatch):
    monkeypatch.setattr(rr, "BinanceSpotWSAdapter", lambda: RealWS())
    monkeypatch.setattr(rr, "PaperAdapter", lambda: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(rr, "BinanceSpotAdapter", RealExec)

    def fake_load():
        os.environ["BINANCE_API_KEY"] = "livek"
        os.environ["BINANCE_API_SECRET"] = "lives"

    monkeypatch.setattr(rr, "load_dotenv", fake_load)

    await rr.run_real(dry_run=False, i_know_what_im_doing=True)

    inst = RealExec.last_instance
    assert inst.api_key == "livek"
    assert inst.api_secret == "lives"
    assert inst.orders[0][:4] == ("BTC/USDT", "buy", "market", 0.001)
