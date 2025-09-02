import pandas as pd
import yaml
import pytest
from tradingbot.core import Account, RiskManager as CoreRiskManager
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
    df = pd.DataFrame({"close": [1, 2, 1]})
    cfg = ScalpPingPongConfig(lookback=2, z_threshold=0.1, volatility_factor=1.0, min_volatility=0.0)
    strat = ScalpPingPong(cfg=cfg)
    sig = strat.on_bar({"window": df, "close": df["close"].iloc[-1], "volatility": 0.0})
    assert sig is not None
    assert sig.limit_price == df["close"].iloc[-1]


@pytest.mark.parametrize(
    "timeframe,closes",
    [
        (1, list(range(100, 115)) + [100]),
        (5, [100, 105, 110, 100]),
    ],
)
def test_scalp_pingpong_generates_trades_across_timeframes(timeframe, closes):
    df = pd.DataFrame({"close": closes})
    cfg = ScalpPingPongConfig(volatility_factor=0.02, min_volatility=0.0)
    strat = ScalpPingPong(cfg=cfg)
    sig = strat.on_bar({"window": df, "timeframe": timeframe})
    assert sig is not None
    assert 0 < sig.strength <= 1.0
