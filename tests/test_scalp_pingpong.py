import math
import types

import pandas as pd
import pytest
import yaml

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.strategies.base import timeframe_to_minutes
from tradingbot.strategies.scalp_pingpong import ScalpPingPong, ScalpPingPongConfig


def test_config_path_overrides(tmp_path):
    data = {"lookback": 25, "z_threshold": 0.3, "volatility_factor": 0.05}
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(data))

    strat = ScalpPingPong(config_path=str(cfg_file))

    assert strat.cfg.lookback == 25
    assert strat.cfg.z_threshold == 0.3
    assert strat.cfg.volatility_factor == 0.05


def test_scalp_pingpong_trailing_stop_uses_atr():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    trade = {
        "side": "buy",
        "entry_price": 100.0,
        "qty": 1.0,
        "stop": 99.0,
        "atr": 1.0,
        "stage": 3,
    }
    rm.update_trailing(trade, 110.0)
    assert trade["stop"] == pytest.approx(110.0 - 2 * trade["atr"])


def test_scalp_pingpong_emits_limit_price():
    closes = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        109,
        108,
        107,
        106,
        105,
        104,
        103,
        102,
        101,
        100,
        99,
        98,
        97,
        96,
        95,
    ]
    df = pd.DataFrame({"close": closes})
    strat = ScalpPingPong()
    sig = strat.on_bar({"window": df, "symbol": "BTCUSDT", "exchange": "paper"})
    assert sig is not None
    assert sig.post_only is True
    assert sig.metadata.get("post_only") is True
    assert 0 < sig.strength <= 1.0
    assert sig.limit_price <= sig.metadata["base_price"]
    assert sig.limit_price < df["close"].iloc[-1]
    partial = sig.metadata.get("partial_take_profit")
    assert isinstance(partial, dict)
    assert partial["mode"] == "scale_out"
    assert sig.metadata.get("max_hold_bars") == 12


@pytest.mark.parametrize(
    "timeframe,closes",
    [
        (1, list(range(100, 115)) + [100]),
        (5, [100, 105, 110, 108, 106, 104, 103, 102, 100]),
    ],
)
def test_scalp_pingpong_generates_trades_across_timeframes(timeframe, closes):
    df = pd.DataFrame({"close": closes})
    cfg = ScalpPingPongConfig(volatility_factor=0.02, min_volatility=0.0)
    strat = ScalpPingPong(cfg=cfg)
    sig = strat.on_bar(
        {
            "window": df,
            "timeframe": timeframe,
            "symbol": "BTCUSDT",
            "exchange": "paper",
        }
    )
    assert sig is not None
    assert 0 < sig.strength <= 1.0


def test_scalp_pingpong_honours_min_notional(caplog):
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                99.0,
                98.0,
                97.0,
                96.0,
                95.0,
                94.0,
                93.0,
                94.0,
                95.0,
                96.0,
                95.0,
                94.0,
                95.0,
                96.0,
                97.0,
                98.0,
            ]
        }
    )
    cfg = ScalpPingPongConfig(
        lookback=2,
        z_threshold=0.1,
        volatility_factor=0.02,
        min_volatility=0.0,
    )

    account = Account(float("inf"), cash=1_000.0)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="test"))
    risk = RiskService(guard, account=account)
    risk.min_notional = 10.0

    strat = ScalpPingPong(cfg=cfg, risk_service=risk)

    bar = {
        "window": df,
        "timeframe": 1,
        "symbol": "BTCUSDT",
        "exchange": "paper",
        "atr": 0.5,
    }

    with caplog.at_level("INFO"):
        sig = strat.on_bar(bar)

    assert sig is not None
    assert sig.strength > 0
    qty = risk.calc_position_size(sig.strength, bar["window"]["close"].iloc[-1])
    assert qty * bar["window"]["close"].iloc[-1] >= risk.min_notional
    assert "orden submínima" not in caplog.text


@pytest.mark.parametrize("timeframe", ["15m", "1h"])
def test_scalp_pingpong_high_timeframe_zscore_breaks_zero(timeframe):
    closes = [
        100.0,
        101.5,
        99.0,
        102.5,
        101.0,
        103.5,
        99.5,
        104.0,
    ]
    df = pd.DataFrame({"close": closes})
    cfg = ScalpPingPongConfig(lookback=15, z_threshold=0.2, volatility_factor=0.02, min_volatility=0.0)
    strat = ScalpPingPong(cfg=cfg)

    tf_minutes = timeframe_to_minutes(timeframe)
    lookback = max(2, math.ceil(cfg.lookback / tf_minutes))
    z = strat._calc_zscore(df["close"], lookback)
    assert z != 0

    sig = strat.on_bar({"window": df, "timeframe": timeframe, "symbol": "BTCUSDT", "exchange": "paper"})
    assert sig is not None
    assert sig.strength > 0


def test_scalp_pingpong_signal_uses_risk_service(monkeypatch):
    closes = [100, 99, 98, 97, 96, 95, 94, 93, 94, 95, 96, 95, 94, 95, 96, 97, 98]
    df = pd.DataFrame({"close": closes})
    account = Account(float("inf"), cash=1_000.0)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    risk = RiskService(guard, account=account)
    risk.account.update_cash(1_000.0)

    calc_calls: list[dict] = []
    orig_calc = risk.calc_position_size

    def spy_calc(self, strength, price, **kwargs):
        calc_calls.append(dict(kwargs))
        return orig_calc(strength, price, **kwargs)

    monkeypatch.setattr(risk, "calc_position_size", types.MethodType(spy_calc, risk))

    stop_calls: list[tuple[float, str, float | None]] = []
    orig_stop = risk.initial_stop

    def spy_stop(self, price, side, atr):
        stop_calls.append((price, side, atr))
        return orig_stop(price, side, atr)

    monkeypatch.setattr(risk, "initial_stop", types.MethodType(spy_stop, risk))

    strat = ScalpPingPong(risk_service=risk)
    bar = {
        "window": df,
        "symbol": "BTCUSDT",
        "exchange": "paper",
        "atr": 0.5,
    }
    sig = strat.on_bar(bar)

    assert sig is not None
    assert calc_calls and calc_calls[0]["clamp"] is True
    assert stop_calls and stop_calls[0][2] == 0.5
    trade = strat.trade
    assert trade is not None
    assert trade["partial_take_profit"]["mode"] == "scale_out"
    assert trade["max_hold"] == 12
    assert trade["strength"] == pytest.approx(sig.strength)


def test_scalp_pingpong_backtest_records_maker_fill():
    pattern = [100, 99, 98, 97, 96, 95, 94, 93, 94, 95, 96, 95, 94, 95, 96, 97, 98]
    closes = [100.0] * 20 + pattern
    length = len(closes)
    data = pd.DataFrame(
        {
            "timestamp": range(length),
            "open": closes,
            "high": [p + 0.6 for p in closes],
            "low": [p - 0.6 for p in closes],
            "close": closes,
            "volume": [1_500] * length,
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("scalp_pingpong", "SYM")],
        window=21,
        latency=1,
        verbose_fills=True,
        risk_pct=0.01,
    )
    result = engine.run()

    fills = [fill for fill in result["fills"] if fill[1] == "order"]
    assert fills, "Expected at least one recorded fill"
    first_fill = fills[0]
    assert first_fill[9] == pytest.approx(0.0)

    orders = result["orders"]
    assert orders
    first_order = orders[0]

    fill_price = float(first_fill[3])
    limit_price = float(first_order["place_price"])
    if first_fill[2] == "buy":
        assert fill_price <= limit_price + 1e-9
    else:
        assert fill_price >= limit_price - 1e-9
