import pandas as pd
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
import importlib, sys, types

# runner_testnet indirectly imports a module missing in tests; inject a stub
binance_ws_stub = types.ModuleType("tradingbot.adapters.binance_ws")
binance_ws_stub.BinanceWSAdapter = object
sys.modules.setdefault("tradingbot.adapters.binance_ws", binance_ws_stub)

from tradingbot.live import runner_testnet as rt
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
        return SimpleNamespace(side="buy", strength=1.0)


class DummyRisk:
    def size(self, side, price, equity, strength, **kwargs):
        return 1.0

    def check_limits(self, price):
        return True

    def add_fill(self, side, qty):
        pass

    def update_correlation(self, pairs, threshold):
        return []

    def update_position(self, exchange, symbol, qty):
        pass


class DummyPG:
    def __init__(self):
        from types import SimpleNamespace
        self.st = SimpleNamespace(venue_positions={})

    def mark_price(self, symbol, px):
        pass

    def soft_cap_decision(self, *a):
        return ("allow", "", None)

    def volatility(self, symbol):
        return 0.0

    def update_position_on_order(self, symbol, side, qty, venue=None):
        pass

    def refresh_usd_caps(self, equity):
        self.equity = equity


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
    monkeypatch.setattr(rt, "BreakoutATR", lambda config_path=None: DummyStrat())
    monkeypatch.setattr(rt, "RiskManager", lambda *a, **k: DummyRisk())
    monkeypatch.setattr(rt, "PortfolioGuard", lambda config: DummyPG())
    monkeypatch.setattr(rt, "DailyGuard", lambda limits, venue: DummyDG())
    monkeypatch.setattr(rt, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rt, "_CAN_PG", False)

    monkeypatch.setattr(
        rt,
        "_SymbolConfig",
        lambda symbol, risk_pct: types.SimpleNamespace(
            symbol=symbol, risk_pct=risk_pct, equity_pct=1.0
        ),
    )
    monkeypatch.setitem(
        rt.ADAPTERS,
        ("bybit", "futures"),
        (lambda: DummyWS(), DummyExec, "bybit_futures_testnet"),
    )

    cfg = rt._SymbolConfig(symbol=normalize("BTC-USDT"), risk_pct=0.0)
    await rt._run_symbol(
        "bybit",
        "futures",
        cfg,
        leverage=5,
        dry_run=False,
        total_cap_pct=1.0,
        per_symbol_cap_pct=0.5,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_pct=0.05,
        daily_max_drawdown_pct=0.05,
        corr_threshold=0.8,
    )

    inst = DummyExec.last_instance
    assert inst.leverage == 5
    assert inst.testnet is True
    assert inst.orders[0][:4] == (normalize("BTC-USDT"), "buy", "market", 1.0)


class DummyExecReal:
    last_instance = None

    def __init__(self, api_key=None, api_secret=None, leverage=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.leverage = leverage
        self.orders = []
        DummyExecReal.last_instance = self

    async def place_order(self, symbol, side, type_, qty, mark_price=None):
        self.orders.append((symbol, side, type_, qty))
        return {"status": "ok"}


@pytest.mark.asyncio
async def test_run_real(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    import importlib, sys
    import tradingbot.config as config
    config = importlib.reload(config)
    sys.modules["tradingbot.config"] = config
    import tradingbot.live.runner_real as rr
    rr = importlib.reload(rr)
    monkeypatch.setattr(rr, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rr, "BreakoutATR", lambda config_path=None: DummyStrat())
    monkeypatch.setattr(rr, "RiskManager", lambda *a, **k: DummyRisk())
    monkeypatch.setattr(rr, "PortfolioGuard", lambda config: DummyPG())
    monkeypatch.setattr(rr, "DailyGuard", lambda limits, venue: DummyDG())
    monkeypatch.setattr(rr, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rr, "_CAN_PG", False)
    monkeypatch.setattr(
        rr,
        "_SymbolConfig",
        lambda symbol, risk_pct: types.SimpleNamespace(
            symbol=symbol, risk_pct=risk_pct, equity_pct=1.0
        ),
    )
    monkeypatch.setitem(
        rr.ADAPTERS,
        ("binance", "spot"),
        (lambda: DummyWS(), DummyExecReal, "binance_spot"),
    )

    cfg = rr._SymbolConfig(symbol=normalize("BTC-USDT"), risk_pct=0.0)
    await rr._run_symbol(
        "binance",
        "spot",
        cfg,
        leverage=1,
        dry_run=False,
        total_cap_pct=1.0,
        per_symbol_cap_pct=0.5,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_pct=0.05,
        daily_max_drawdown_pct=0.05,
        corr_threshold=0.8,
    )

    inst = DummyExecReal.last_instance
    assert inst.api_key == "k"
    assert inst.orders[0][:4] == (normalize("BTC-USDT"), "buy", "market", 1.0)


@pytest.mark.asyncio
async def test_real_requires_flag(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "x")
    monkeypatch.setenv("BINANCE_API_SECRET", "y")
    import importlib, sys
    import tradingbot.config as config
    config = importlib.reload(config)
    sys.modules["tradingbot.config"] = config
    import tradingbot.live.runner_real as rr
    rr = importlib.reload(rr)
    with pytest.raises(ValueError):
        await rr.run_live_real(symbols=["BTC/USDT"], i_know_what_im_doing=False)


class DummyExec2(DummyExec):
    pass


@pytest.mark.asyncio
async def test_okx_futures_order(monkeypatch):
    monkeypatch.setattr(rt, "BarAggregator", DummyAgg)
    monkeypatch.setattr(rt, "BreakoutATR", lambda config_path=None: DummyStrat())
    monkeypatch.setattr(rt, "RiskManager", lambda *a, **k: DummyRisk())
    monkeypatch.setattr(rt, "PortfolioGuard", lambda config: DummyPG())
    monkeypatch.setattr(rt, "DailyGuard", lambda limits, venue: DummyDG())
    monkeypatch.setattr(rt, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rt, "_CAN_PG", False)

    monkeypatch.setattr(
        rt,
        "_SymbolConfig",
        lambda symbol, risk_pct: types.SimpleNamespace(
            symbol=symbol, risk_pct=risk_pct, equity_pct=1.0
        ),
    )
    monkeypatch.setitem(
        rt.ADAPTERS,
        ("okx", "futures"),
        (lambda: DummyWS(), DummyExec2, "okx_futures_testnet"),
    )

    cfg = rt._SymbolConfig(symbol=normalize("BTC-USDT"), risk_pct=0.0)
    await rt._run_symbol(
        "okx",
        "futures",
        cfg,
        leverage=7,
        dry_run=False,
        total_cap_pct=1.0,
        per_symbol_cap_pct=0.5,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_pct=0.05,
        daily_max_drawdown_pct=0.05,
        corr_threshold=0.8,
    )

    inst = DummyExec2.last_instance
    assert inst.leverage == 7
    assert inst.testnet is True
    assert inst.orders[0][:4] == (normalize("BTC-USDT"), "buy", "market", 1.0)
