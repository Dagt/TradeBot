import math
import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.execution.order_types import Order
from tradingbot.strategies.base import Signal
from tradingbot.strategies.breakout_atr import BreakoutATR
import tradingbot.strategies.breakout_atr as breakout_mod


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
        lambda self, *args, **kwargs: (0.0, 0.0),
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
        lambda self, *args, **kwargs: (0.0, 0.0),
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


def _synthetic_breakout_window(two_closes: bool) -> pd.DataFrame:
    base_opens = [100.0 + 0.15 * i for i in range(10)]
    base_closes = [open_ + 0.08 for open_ in base_opens]
    base_highs = [close + 0.12 for close in base_closes]
    base_lows = [open_ - 0.12 for open_ in base_opens]

    if two_closes:
        prev_open = 101.6
        prev_close = 101.95
        final_open = 102.0
        final_close = 102.9
        final_low = min(final_open, final_close) - 0.25
        final_high = final_close + 0.2
        prev_high = prev_close + 0.15
        prev_low = prev_open - 0.2
    else:
        prev_open = 101.2
        prev_close = 101.3
        final_open = 101.4
        final_close = 102.5
        final_low = min(final_open, final_close) - 0.35
        final_high = final_close + 0.2
        prev_high = prev_close + 0.15
        prev_low = prev_open - 0.2

    opens = base_opens + [prev_open, final_open]
    closes = base_closes + [prev_close, final_close]
    highs = base_highs + [prev_high, final_high]
    lows = base_lows + [prev_low, final_low]
    volume = [5] * len(opens)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
    )


def _synthetic_backtest_window() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total = 40
    confirmed_indices = {32, 36}
    for idx in range(total):
        base_upper = 100.0 + 0.05 * idx
        volume = 12.0
        if idx in confirmed_indices:
            open_ = base_upper + 0.1
            close = base_upper + 0.9
            volume = 18.0
        elif idx + 1 in confirmed_indices:
            open_ = base_upper - 0.05
            close = base_upper + 0.3
        else:
            open_ = base_upper - 0.4
            close = base_upper - 0.2
        high = max(open_, close) + 0.15
        low = min(open_, close) - 0.15
        rows.append(
            {
                "timestamp": float(idx),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("two_closes", [False, True])
def test_breakout_atr_emits_signals_for_marginal_and_confirmed_breakouts(
    monkeypatch: pytest.MonkeyPatch, two_closes: bool
) -> None:
    def fake_atr(df: pd.DataFrame, period: int) -> pd.Series:
        base = 0.6
        step = 0.015
        values = [base + step * i for i in range(len(df))]
        return pd.Series(values, index=df.index)

    upper_values = [
        100.3,
        100.45,
        100.6,
        100.75,
        100.9,
        101.05,
        101.2,
        101.35,
        101.5,
        101.6,
        101.2,
        101.7,
    ]

    def fake_kc(
        df: pd.DataFrame, ema_n: int, atr_n: int, mult: float
    ) -> tuple[pd.Series, pd.Series]:
        upper = pd.Series(upper_values[: len(df)], index=df.index)
        lower = upper - 1.4
        return upper, lower

    monkeypatch.setattr(breakout_mod, "atr", fake_atr)
    monkeypatch.setattr(breakout_mod, "keltner_channels", fake_kc)
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_allows_side",
        staticmethod(lambda side, regime, threshold: True),
    )
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_threshold",
        lambda self, *args, **kwargs: (0.0, 0.0),
    )
    monkeypatch.setattr(
        BreakoutATR,
        "_extreme_penetration_cap",
        lambda self, *args, **kwargs: 999.0,
    )

    strat = BreakoutATR(ema_n=3, atr_n=3, volume_factor=0.0)
    df = _synthetic_breakout_window(two_closes)
    bar = {"window": df, "timeframe": "1m", "symbol": "FAST/USDT"}

    signal = strat.on_bar(bar)

    assert signal is not None
    assert signal.metadata["breakout_penetration"] >= signal.metadata["breakout_penetration_threshold"]
    if two_closes:
        assert signal.metadata["breakout_two_closes_beyond"] is True
        assert signal.metadata["breakout_single_body"] is False
    else:
        assert signal.metadata["breakout_single_body"] is True
        assert signal.metadata["breakout_two_closes_beyond"] is False


def test_breakout_atr_backtest_counts_fills_for_marginal_and_confirmed_breakouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_atr(df: pd.DataFrame, period: int) -> pd.Series:
        base = 0.5
        step = 0.01
        values = [base + step * i for i in range(len(df))]
        return pd.Series(values, index=df.index)

    def fake_kc(
        df: pd.DataFrame, ema_n: int, atr_n: int, mult: float
    ) -> tuple[pd.Series, pd.Series]:
        upper = pd.Series([100.0 + 0.05 * i for i in range(len(df))], index=df.index)
        lower = upper - 1.4
        return upper, lower

    monkeypatch.setattr(breakout_mod, "atr", fake_atr)
    monkeypatch.setattr(breakout_mod, "keltner_channels", fake_kc)
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_allows_side",
        staticmethod(lambda side, regime, threshold: True),
    )
    monkeypatch.setattr(
        BreakoutATR,
        "_regime_threshold",
        lambda self, *args, **kwargs: (0.0, 0.0),
    )
    monkeypatch.setattr(
        BreakoutATR,
        "_extreme_penetration_cap",
        lambda self, *args, **kwargs: 999.0,
    )

    class DummyRQ:
        def update(self, value):
            if value is None or not math.isfinite(value):
                return value
            return value * 0.9

    monkeypatch.setattr(
        breakout_mod.RollingQuantileCache,
        "get",
        lambda self, symbol, name, *, window, q, min_periods=None: DummyRQ(),
    )

    data = _synthetic_backtest_window()
    warmup = 36
    engine = EventDrivenBacktestEngine(
        {"FAST/USDT": data},
        [("breakout_atr", "FAST/USDT")],
        timeframes={"FAST/USDT": "3m"},
        window=warmup,
        verbose_fills=True,
        risk_pct=0.02,
    )
    result = engine.run()

    assert result["fill_count"] >= 2
    assert len(result["fills"]) >= 2
