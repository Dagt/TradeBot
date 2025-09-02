from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tradingbot.cli import main as cli_main
from tradingbot.backtest import event_engine as ev_module
from tradingbot.backtesting.engine import EventDrivenBacktestEngine, STRATEGIES
from tradingbot.config.hydra_conf import load_config
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
        timeframe="3m",
        capital=0.0,
        risk_pct=0.0,
        fee_bps=5.0,
        slippage_bps=1.0,
        verbose_fills=False,
    )
    return captured["engine"]


def _load_cfg():
    cfg = load_config()
    return OmegaConf.to_container(getattr(cfg, "exchange_configs", {}), resolve=True)


def test_backtest_db_futures_config(monkeypatch):
    eng = _run_backtest(monkeypatch, "binance_futures")
    cfg = _load_cfg()["binance_futures"]
    assert eng.exchange_mode["binance_futures"] == cfg["market_type"]
    assert eng.exchange_fees["binance_futures"].maker_fee == pytest.approx(
        cfg["maker_fee_bps"] / 10000.0
    )
    assert eng.exchange_fees["binance_futures"].taker_fee == pytest.approx(
        cfg["taker_fee_bps"] / 10000.0
    )
    assert eng.exchange_tick_size["binance_futures"] == cfg["tick_size"]
    rs = eng.risk[("dummy", "BTCUSDT")]
    assert rs.allow_short is True


def test_backtest_db_spot_config(monkeypatch):
    eng = _run_backtest(monkeypatch, "binance_spot")
    cfg = _load_cfg()["binance_spot"]
    assert eng.exchange_mode["binance_spot"] == cfg["market_type"]
    assert eng.exchange_fees["binance_spot"].maker_fee == pytest.approx(
        cfg["maker_fee_bps"] / 10000.0
    )
    assert eng.exchange_fees["binance_spot"].taker_fee == pytest.approx(
        cfg["taker_fee_bps"] / 10000.0
    )
    assert eng.exchange_tick_size["binance_spot"] == cfg["tick_size"]
    rs = eng.risk[("dummy", "BTCUSDT")]
    assert rs.allow_short is False


def test_backtest_db_normalizes_symbol(monkeypatch):
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
    ]

    def fake_select_bars(*args, **kwargs):
        captured["symbol"] = kwargs["symbol"]
        return rows

    monkeypatch.setattr(ts_module, "select_bars", fake_select_bars)

    class DummyStrategy:
        name = "dummy"

        def on_bar(self, bar):
            return SimpleNamespace(side="buy", strength=1.0)

    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)

    cli_main.backtest_db(
        venue="binance_spot",
        symbol="BTC/USDT",
        strategy="dummy",
        start="2021-01-01",
        end="2021-01-02",
        timeframe="3m",
        capital=0.0,
        risk_pct=0.0,
        fee_bps=5.0,
        slippage_bps=1.0,
        verbose_fills=False,
    )

    assert captured["symbol"] == "BTCUSDT"
