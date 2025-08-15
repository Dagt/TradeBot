import logging
from pathlib import Path
import pandas as pd
from ..strategies import STRATEGIES
from ..risk.manager import RiskManager

log = logging.getLogger(__name__)

def run_backtest_csv(csv_path: str, symbol: str = "BTC/USDT", strategy: str = "breakout_atr"):
    path = Path(csv_path)
    df = pd.read_csv(path)
    # Expected columns: timestamp, open, high, low, close, volume
    equity = 0.0
    pos = 0.0
    strat_cls = STRATEGIES.get(strategy)
    if strat_cls is None:
        raise ValueError(f"unknown strategy: {strategy}")
    strat = strat_cls()
    risk = RiskManager(max_pos=1.0)

    window = 120  # bars for features
    fills = []

    for i in range(len(df)):
        if i < window:
            continue
        win = df.iloc[i-window:i].copy()
        bar = {"window": win}
        sig = strat.on_bar(bar)
        if sig is None:
            continue
        delta = risk.size(sig.side, sig.strength)
        # Fill at next close as simplificación
        px = df["close"].iloc[i]
        if abs(delta) > 1e-9:
            equity -= delta * px  # cash change
            pos += delta
            fills.append((df["timestamp"].iloc[i], px, delta))
    # Liquidar posición
    last_px = df["close"].iloc[-1]
    equity += pos * last_px
    result = {"equity": equity, "fills": fills, "last_px": last_px}
    log.info("Backtest result: %s", result)
    return result
