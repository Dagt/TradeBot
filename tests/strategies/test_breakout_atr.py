import pandas as pd
import pytest

from tradingbot.execution.order_types import Order
from tradingbot.strategies.base import Signal
from tradingbot.strategies.breakout_atr import BreakoutATR


def _choppy_guard_window() -> pd.DataFrame:
    base = 200.0
    closes = [0.05] * 26 + [-0.2, -0.2, -0.2, 1.6, 2.4]
    highs = [c + 0.2 for c in closes]
    lows = [-0.5] * len(closes)
    for idx in (26, 27, 28):
        highs[idx] = 1.6
        lows[idx] = -0.8
    highs[-2] = closes[-2] + 0.2
    highs[-1] = closes[-1] + 0.3
    lows[-2] = -0.4
    lows[-1] = -0.4
    return pd.DataFrame(
        {
            "open": base,
            "close": [base + c for c in closes],
            "high": [base + h for h in highs],
            "low": [base + l for l in lows],
            "volume": 15.0,
        }
    )


def _choppy_reset_window() -> pd.DataFrame:
    base = 200.0
    closes = [0.05] * 24 + [0.35, 0.75, 1.3, 1.9, 2.45, 3.1]
    highs: list[float] = []
    lows: list[float] = []
    last_index = len(closes) - 1
    for idx, close in enumerate(closes):
        if idx < 24:
            highs.append(close + 0.18)
            lows.append(-0.5)
        elif idx < last_index:
            highs.append(close + 0.05)
            lows.append(-0.4)
        else:
            highs.append(close + 0.25)
            lows.append(-0.4)
    return pd.DataFrame(
        {
            "open": base,
            "close": [base + c for c in closes],
            "high": [base + h for h in highs],
            "low": [base + l for l in lows],
            "volume": 15.0,
        }
    )


@pytest.fixture
def breakout_wick_window() -> pd.DataFrame:
    return _choppy_guard_window()


@pytest.fixture
def breakout_reset_window() -> pd.DataFrame:
    return _choppy_reset_window()


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


def test_breakout_atr_maker_patience_preserves_initial_quote():
    strat = BreakoutATR()
    symbol = "BTC/USDT"
    sig = Signal("buy", 1.0)
    base_price = 100.0
    sig.limit_price = base_price
    sig.metadata.update(
        {
            "base_price": base_price,
            "limit_offset": 3.0,
            "initial_offset": 0.0,
            "offset_step": 1.0,
            "maker_initial_offset": 1.0,
            "maker_patience": 2,
            "chase": True,
        }
    )
    strat._last_signal = {symbol: sig}
    order = Order(symbol=symbol, side="buy", type_="limit", qty=1.0, price=base_price)
    res = {"price": base_price}

    strat._requote_attempts = {(symbol, "buy"): 1}
    strat._reprice_order(order, res)
    assert order.price == pytest.approx(base_price)

    strat._requote_attempts[(symbol, "buy")] = 2
    strat._reprice_order(order, res)
    assert order.price == pytest.approx(base_price)

    strat._requote_attempts[(symbol, "buy")] = 3
    strat._reprice_order(order, res)
    assert order.price == pytest.approx(base_price + 1.0)

    strat._requote_attempts[(symbol, "buy")] = 4
    strat._reprice_order(order, res)
    assert order.price == pytest.approx(base_price + 2.0)

    strat._requote_attempts[(symbol, "buy")] = 6
    strat._reprice_order(order, res)
    # limit_offset caps the progression even when attempts keep increasing
    assert order.price == pytest.approx(base_price + 3.0)


def test_breakout_atr_maker_patience_applies_to_sell_orders():
    strat = BreakoutATR()
    symbol = "ETH/USDT"
    sig = Signal("sell", 1.0)
    base_price = 50.0
    sig.limit_price = base_price
    sig.metadata.update(
        {
            "base_price": base_price,
            "limit_offset": 2.0,
            "initial_offset": 0.0,
            "offset_step": 0.5,
            "maker_initial_offset": 0.8,
            "maker_patience": 1,
            "chase": True,
        }
    )
    strat._last_signal = {symbol: sig}
    order = Order(symbol=symbol, side="sell", type_="limit", qty=1.0, price=base_price)
    res = {"price": base_price}

    strat._requote_attempts = {(symbol, "sell"): 1}
    strat._reprice_order(order, res)
    # First attempt keeps the order at the maker ask
    assert order.price == pytest.approx(base_price)

    strat._requote_attempts[(symbol, "sell")] = 2
    strat._reprice_order(order, res)
    # First offset activates after patience lapses
    assert order.price == pytest.approx(base_price - 0.8)

    strat._requote_attempts[(symbol, "sell")] = 5
    strat._reprice_order(order, res)
    # Step increments remain capped by the limit offset
    assert order.price == pytest.approx(base_price - 2.0)


def test_breakout_atr_skips_breakout_after_nearby_extremes(
    breakout_wick_window: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_allows_side",
        staticmethod(lambda side, regime, threshold: True),
    )
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_threshold",
        lambda self, *args, **kwargs: 0.0,
    )
    strat = BreakoutATR(ema_n=10, atr_n=10, volume_factor=0.0)
    strat.min_regime = 0.0
    strat.max_regime = 0.0
    bar = {
        "window": breakout_wick_window,
        "timeframe": "3m",
        "symbol": "TEST/USDT",
    }

    signal = strat.on_bar(bar)

    assert signal is None
    assert strat.last_prior_extreme_penetration >= strat.last_extreme_penetration_cap


def test_breakout_atr_accepts_breakout_after_consolidation(
    breakout_reset_window: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_allows_side",
        staticmethod(lambda side, regime, threshold: True),
    )
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_threshold",
        lambda self, *args, **kwargs: 0.0,
    )
    strat = BreakoutATR(ema_n=10, atr_n=10, volume_factor=0.0)
    strat.min_regime = 0.0
    strat.max_regime = 0.0
    bar = {
        "window": breakout_reset_window,
        "timeframe": "3m",
        "symbol": "TEST/USDT",
    }

    signal = strat.on_bar(bar)

    assert signal is not None
    assert signal.side == "buy"
    assert (
        signal.metadata["breakout_prior_extreme_penetration"]
        < signal.metadata["breakout_extreme_cap"]
    )
