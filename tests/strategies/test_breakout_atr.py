import pandas as pd

from tradingbot.strategies.breakout_atr import BreakoutATR


def _downtrend_with_bull_breakout(length: int = 60) -> pd.DataFrame:
    base = 200.0
    trend = -1.2
    opens = [base + i * trend for i in range(length)]
    closes = [o - 0.6 for o in opens]
    highs = [max(o, c) + 0.8 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.8 for o, c in zip(opens, closes)]
    volume = [5.0] * length

    opens[-1] = opens[-2] + 2.0
    closes[-1] = closes[-2] + 6.0
    highs[-1] = closes[-1] + 1.0
    lows[-1] = min(opens[-1], closes[-1]) - 1.0

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


def test_breakout_atr_rejects_bull_breakout_in_bearish_regime():
    df = _downtrend_with_bull_breakout()
    strat = BreakoutATR(ema_n=10, atr_n=10, min_regime=0.15, max_regime=0.3, volume_factor=0.0)
    bar = {"window": df, "volatility": 0.0, "timeframe": "1m", "symbol": "TEST/USDT"}

    signal = strat.on_bar(bar)

    assert signal is None
    assert strat.last_regime < 0
    assert strat.last_regime_threshold > 0
    assert abs(strat.last_regime) >= strat.last_regime_threshold


def test_breakout_atr_max_hold_scales_with_strength_and_regime():
    strat = BreakoutATR()

    base_hold = strat._max_hold_bars(3.0, strength=0.55, regime_abs=1.0)
    strong_hold = strat._max_hold_bars(3.0, strength=0.9, regime_abs=1.8)
    slow_hold = strat._max_hold_bars(20.0, strength=0.6, regime_abs=1.2)
    slow_strong = strat._max_hold_bars(20.0, strength=0.95, regime_abs=1.9)

    assert strong_hold > base_hold >= 10
    assert strong_hold <= 160
    assert slow_strong >= slow_hold


def test_breakout_atr_partial_take_profit_adapts_to_timeframe_and_strength():
    strat = BreakoutATR()

    fast_high = strat._partial_take_profit_config(3.0, 0.9, 1.6)
    fast_low = strat._partial_take_profit_config(3.0, 0.25, 0.8)
    slow_high = strat._partial_take_profit_config(30.0, 0.85, 1.4)

    assert fast_high["atr_multiple"] >= 2.0
    assert fast_high["qty_pct"] < 0.2
    assert fast_low["qty_pct"] <= 0.1
    assert fast_low["qty_pct"] >= 0.0
    assert slow_high["atr_multiple"] > fast_low["atr_multiple"]
    assert slow_high["qty_pct"] >= fast_low["qty_pct"]
