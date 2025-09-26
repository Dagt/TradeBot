import pandas as pd

from tradingbot.data.features import keltner_channels
from tradingbot.strategies.breakout_atr import BreakoutATR


def _make_bearish_breakout(
    rows: int = 200, slope: float = -0.35, breakout_jump: float = 18.0
) -> pd.DataFrame:
    base = 100.0
    closes: list[float] = []
    for idx in range(rows):
        closes.append(base + slope * idx)

    # Accentuate the downtrend before the breakout so the regime stays negative.
    for offset, idx in enumerate(range(rows - 6, rows - 1)):
        closes[idx] -= (offset + 1) * 1.5

    closes[-1] = closes[-2] + breakout_jump

    highs = [c + 0.6 for c in closes]
    lows = [c - 0.6 for c in closes]
    opens = [closes[0]] + closes[:-1]
    volume = [75.0] * rows

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
    )


def test_breakout_atr_rejects_counter_trend_bull_breakout():
    strat = BreakoutATR(timeframe="5m", volume_factor=0.0, min_regime=0.18, max_regime=0.55)
    df = _make_bearish_breakout()
    bar = {"window": df, "timeframe": "5m", "symbol": "X/USDT", "volatility": 0.0}

    sig = strat.on_bar(bar)

    assert sig is None
    assert strat.last_regime < 0.0
    assert abs(strat.last_regime) >= strat.last_regime_threshold
    assert strat.last_regime_threshold > 0.0

    tf_mult = strat._tf_multiplier("5m")
    ema_n = strat._ema_period(tf_mult)
    atr_n = strat._atr_period(tf_mult)
    upper, _ = keltner_channels(df, ema_n, atr_n, strat.mult)

    assert df["close"].iloc[-1] > upper.iloc[-1]
    alignment = strat._regime_alignment(strat.last_regime, strat.last_regime_threshold, "buy")
    assert alignment == 0.0
