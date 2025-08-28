from types import SimpleNamespace

from tradingbot.config import settings
from tradingbot.cli import main as cli_main
from tradingbot.backtest import event_engine as ev_module
from tradingbot.backtesting.engine import EventDrivenBacktestEngine, STRATEGIES
from tradingbot.storage import timescale as ts_module


def _run_backtest(monkeypatch, venue):
    captured = {}

    class DummyEngine(EventDrivenBacktestEngine):
        def run(self, fills_csv=None):
            captured["engine"] = self
            return {"equity": 0.0, "orders": []}

    monkeypatch.setattr(ev_module, "EventDrivenBacktestEngine", DummyEngine)

    class DummyDB:
        def dispose(self):
            pass

    monkeypatch.setattr(ts_module, "get_engine", lambda: DummyDB())
    rows = [
        {"ts": 0, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0},
        {"ts": 1, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0},
    ]
    monkeypatch.setattr(ts_module, "select_bars", lambda *args, **kwargs: rows)

    class DummyStrategy:
        name = "dummy"

        def on_bar(self, bar):
            return SimpleNamespace(side="buy", strength=1.0)

    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)

    cli_main.backtest_db(
        venue=venue,
        symbol="BTC/USDT",
        strategy="dummy",
        start="2021-01-01",
        end="2021-01-02",
        timeframe="1m",
        capital=0.0,
        risk_pct=0.0,
    )
    return captured["engine"]


def test_backtest_db_futures_config(monkeypatch):
    eng = _run_backtest(monkeypatch, "binance_futures")
    fee = settings.binance_futures_taker_fee_bps / 10000
    assert eng.exchange_mode["binance_futures"] == "perp"
    assert eng.exchange_fees["binance_futures"].fee == fee


def test_backtest_db_spot_config(monkeypatch):
    eng = _run_backtest(monkeypatch, "binance_spot")
    fee = settings.binance_spot_taker_fee_bps / 10000
    assert eng.exchange_mode["binance_spot"] == "spot"
    assert eng.exchange_fees["binance_spot"].fee == fee
