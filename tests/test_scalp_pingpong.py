import yaml
from tradingbot.strategies.scalp_pingpong import ScalpPingPong


def test_config_path_overrides(tmp_path):
    data = {"lookback": 25, "z_threshold": 0.3, "trailing_stop_bps": 7.0}
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(data))

    strat = ScalpPingPong(config_path=str(cfg_file))

    assert strat.cfg.lookback == 25
    assert strat.cfg.z_threshold == 0.3
    assert strat.cfg.trailing_stop_bps == 7.0
