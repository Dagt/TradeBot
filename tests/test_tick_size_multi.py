import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.live import runner_testnet as rt
from tradingbot.core import normalize


class DummyWS:
    async def stream_trades(self, symbol):
        yield {"ts": datetime.now(timezone.utc), "price": 100.0, "qty": 1.0}


class DummyAgg:
    def __init__(self, timeframe="1m"):
        pass

    def on_trade(self, ts, px, qty):
        return SimpleNamespace(c=px)

    def last_n_bars_df(self, n):
        return pd.DataFrame({"c": [1.0] * 200})


class DummyStrat:
    def on_bar(self, ctx):
        return SimpleNamespace(side="buy", strength=1.0, reduce_only=False, limit_price=None)


class DummyRisk:
    def __init__(self):
        self.min_order_qty = 0.0
        self.rm = SimpleNamespace(allow_short=True)
        self.account = SimpleNamespace(
            current_exposure=lambda symbol: (0.0, 0.0),
            update_open_order=lambda symbol, side, qty: None,
            open_orders={},
        )

    def mark_price(self, symbol, px):
        pass

    def get_trade(self, symbol):
        return None

    def check_order(self, symbol, side, price, strength=1.0, **_):
        return True, "", 1.0

    def register_order(self, symbol, notional):
        return True

    def on_fill(self, *a, **k):
        pass

    def daily_mark(self, broker, symbol, price, delta_rpnl):
        return False, ""

    def purge(self, symbols):
        pass

    def update_correlation(self, *a, **k):
        return []


class DummyPG:
    def refresh_usd_caps(self, equity):
        pass


class DummyDG:
    def check_halt(self, broker=None):
        return False, ""

    def on_mark(self, *a, **k):
        pass


class DummyBroker:
    def __init__(self, fee_bps=0):
        self.account = SimpleNamespace(
            current_exposure=lambda symbol: (0.0, 0.0),
            update_open_order=lambda symbol, side, qty: None,
        )
        self.state = SimpleNamespace(realized_pnl=0.0, last_px={}, order_book={})

    def update_last_price(self, symbol, px):
        self.state.last_px[symbol] = px

    def equity(self, *_):
        return 1000.0


TICK_MAP = {
    normalize("BTC-USDT"): 0.1,
    normalize("ETH-USDT"): 0.01,
}


class DummyExec:
    last_instances: list = []

    def __init__(self, **kwargs):
        self.orders: list = []
        DummyExec.last_instances.append(self)
        self.meta = types.SimpleNamespace(
            client=types.SimpleNamespace(symbols=["BTC/USDT", "ETH/USDT"]),
            rules_for=lambda sym: SimpleNamespace(
                price_step=TICK_MAP[normalize(sym)], qty_step=1e-9
            ),
        )

    async def place_order(self, symbol, side, type_, qty, price=None, **_):
        self.orders.append((symbol, price))
        return {"status": "ok", "qty": qty, "price": price}


class DummyCorr:
    def get_correlations(self):
        return []


@pytest.mark.asyncio
async def test_tick_size_per_symbol(monkeypatch):
    monkeypatch.setattr(rt, "BarAggregator", DummyAgg)
    monkeypatch.setitem(rt.STRATEGIES, "breakout_atr", lambda **_: DummyStrat())
    monkeypatch.setattr(rt, "RiskService", lambda *a, **k: DummyRisk())
    monkeypatch.setattr(rt, "PortfolioGuard", lambda config: DummyPG())
    monkeypatch.setattr(rt, "DailyGuard", lambda limits, venue: DummyDG())
    monkeypatch.setattr(rt, "PaperAdapter", DummyBroker)
    monkeypatch.setattr(rt, "CorrelationService", lambda *a, **k: DummyCorr())
    monkeypatch.setattr(rt, "_CAN_PG", False)
    monkeypatch.setattr(rt, "CANCELS", types.SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(
        rt,
        "settings",
        types.SimpleNamespace(
            limit_offset_ticks=1,
            limit_expiry_sec=0,
            maker_fee_bps=0.0,
            passive_rebate_bps=0.0,
            requote_attempts=1,
        ),
    )
    monkeypatch.setitem(
        rt.ADAPTERS,
        ("binance", "spot"),
        (lambda: DummyWS(), DummyExec, "binance_spot_testnet"),
    )

    DummyExec.last_instances = []
    cfg1 = rt._SymbolConfig(symbol="BTC/USDT", risk_pct=0.0)
    cfg2 = rt._SymbolConfig(symbol="ETH/USDT", risk_pct=0.0)

    await rt._run_symbol(
        "binance",
        "spot",
        cfg1,
        leverage=1,
        dry_run=False,
        total_cap_pct=1.0,
        per_symbol_cap_pct=1.0,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_pct=0.05,
        daily_max_drawdown_pct=0.05,
        corr_threshold=0.8,
        strategy_name="breakout_atr",
    )

    await rt._run_symbol(
        "binance",
        "spot",
        cfg2,
        leverage=1,
        dry_run=False,
        total_cap_pct=1.0,
        per_symbol_cap_pct=1.0,
        soft_cap_pct=0.1,
        soft_cap_grace_sec=30,
        daily_max_loss_pct=0.05,
        daily_max_drawdown_pct=0.05,
        corr_threshold=0.8,
        strategy_name="breakout_atr",
    )

    order1 = DummyExec.last_instances[0].orders[0]
    order2 = DummyExec.last_instances[1].orders[0]
    assert order1[0] == normalize("BTC-USDT")
    assert order2[0] == normalize("ETH-USDT")
    assert order1[1] == pytest.approx(100.0 - TICK_MAP[normalize("BTC-USDT")])
    assert order2[1] == pytest.approx(100.0 - TICK_MAP[normalize("ETH-USDT")])

