import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_command_terminates_quickly(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
backtest:
  data: data/examples/btcusdt_3m.csv
  symbol: BTC/USDT
  strategy: breakout_atr
walk_forward:
  data: data/examples/btcusdt_3m.csv
  symbol: BTC/USDT
  strategy: breakout_atr
  param_grid: {}
"""
    )
    cmd = [sys.executable, "-m", "tradingbot.cli", "cfg-validate", str(cfg)]
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "TRADINGBOT_SKIP_NTP_CHECK": "1",
    }
    start = time.monotonic()
    subprocess.run(cmd, env=env, check=True, timeout=10)
    elapsed = time.monotonic() - start
    assert elapsed < 5
