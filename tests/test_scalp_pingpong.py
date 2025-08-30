import yaml
import pytest
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.strategies.scalp_pingpong import ScalpPingPong


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
