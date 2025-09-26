import math
import pandas as pd
import numpy as np
import yaml
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.order_flow import OrderFlow
from tradingbot.strategies.mean_rev_ofi import MeanRevOFI
from tradingbot.strategies.breakout_vol import BreakoutVol
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
import pytest
from hypothesis import given, strategies as st, settings
from tradingbot.data.features import keltner_channels, atr
from tradingbot.execution.order_types import Order


def _make_ohlcv(rows: int, *, slope: float = 0.2, noise: float = 0.0) -> pd.DataFrame:
    base = 100.0
    closes = [base + slope * i + noise * ((-1) ** i) for i in range(rows)]
    highs = [c + 0.3 for c in closes]
    lows = [c - 0.3 for c in closes]
    opens = [base + slope * i for i in range(rows)]
    volume = [50.0] * rows
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


def _make_breakout_df(
    rows: int,
    *,
    slope: float,
    noise: float,
    jump: float,
    period: int,
) -> pd.DataFrame:
    df = _make_ohlcv(rows, slope=slope, noise=noise).copy()
    for idx in range(period - 1, rows, period):
        df.loc[df.index[idx], "close"] += jump
        df.loc[df.index[idx], "high"] = df.loc[df.index[idx], "close"] + abs(jump) * 0.4 + 0.3
    return df


def _count_signals(
    strat: BreakoutATR, df: pd.DataFrame, timeframe: str, symbol: str
) -> tuple[int, list[float], float]:
    signals = 0
    thresholds: list[float] = []
    last_threshold = float("nan")
    for end in range(max(strat.ema_n, strat.atr_n) + 5, len(df) + 1):
        window = df.iloc[:end]
        bar = {"window": window, "timeframe": timeframe, "symbol": symbol, "volatility": 0.0}
        sig = strat.on_bar(bar)
        if math.isfinite(strat.last_regime_threshold):
            last_threshold = strat.last_regime_threshold
        if sig is not None:
            signals += 1
            thresholds.append(sig.metadata["regime_threshold"])
    return signals, thresholds, last_threshold


def test_breakout_atr_timeframe_parameters_scale():
    strat = BreakoutATR(timeframe="3m")
    fast_tf = strat._ema_period(1)
    slow_tf = strat._ema_period(60)
    assert fast_tf < slow_tf

    fast_atr = strat._atr_period(1)
    slow_atr = strat._atr_period(60)
    assert fast_atr < slow_atr

    fast_stop = strat._stop_multiplier(1)
    slow_stop = strat._stop_multiplier(60)
    assert fast_stop < slow_stop

    fast_hold = strat._max_hold_bars(1)
    slow_hold = strat._max_hold_bars(60)
    assert fast_hold < slow_hold


def test_breakout_atr_regime_filters_sideways():
    strat = BreakoutATR(timeframe="5m")
    df = _make_ohlcv(120, slope=0.0, noise=0.002)
    bar = {"window": df, "timeframe": "5m", "symbol": "X", "volatility": 0.0}
    sig = strat.on_bar(bar)
    assert sig is None
    assert abs(strat.last_regime) < strat.last_regime_threshold
    assert strat.last_regime_threshold >= strat.min_regime


def test_breakout_atr_signal_count_scales_with_timeframe():
    df = _make_breakout_df(200, slope=0.12, noise=0.0005, jump=1.5, period=25)
    strat_fast = BreakoutATR(
        timeframe="1m", volume_factor=0.0, min_regime=0.18, max_regime=0.55
    )
    fast_count, fast_thresholds, fast_last = _count_signals(strat_fast, df, "1m", "FAST_TF")

    strat_slow = BreakoutATR(
        timeframe="15m", volume_factor=0.0, min_regime=0.18, max_regime=0.55
    )
    slow_count, slow_thresholds, slow_last = _count_signals(strat_slow, df, "15m", "SLOW_TF")

    assert fast_count != slow_count
    assert fast_count > slow_count
    assert slow_count > 0
    assert slow_thresholds
    if fast_thresholds:
        avg_fast = sum(fast_thresholds) / len(fast_thresholds)
        avg_slow = sum(slow_thresholds) / len(slow_thresholds)
        assert avg_slow <= avg_fast
    else:
        # Fast timeframe is stricter and may filter all breakouts.
        assert fast_count == 0
    assert fast_last >= slow_last or math.isnan(fast_last)


def test_breakout_atr_choppy_noise_reduces_signals():
    trend_df = _make_breakout_df(220, slope=0.18, noise=0.0005, jump=1.8, period=30)
    strat_trend = BreakoutATR(
        timeframe="3m", volume_factor=0.0, min_regime=0.18, max_regime=0.6
    )
    trend_count, trend_thresholds, trend_last = _count_signals(
        strat_trend, trend_df, "3m", "TREND_SIG"
    )

    choppy_df = _make_breakout_df(220, slope=0.02, noise=0.01, jump=1.8, period=30)
    strat_choppy = BreakoutATR(
        timeframe="3m", volume_factor=0.0, min_regime=0.18, max_regime=0.6
    )
    choppy_count, choppy_thresholds, choppy_last = _count_signals(
        strat_choppy, choppy_df, "3m", "CHOP_SIG"
    )

    assert trend_count > 0
    assert choppy_count < trend_count
    assert choppy_last >= trend_last
    if choppy_thresholds:
        avg_choppy = sum(choppy_thresholds) / len(choppy_thresholds)
        avg_trend = sum(trend_thresholds) / len(trend_thresholds)
        assert avg_choppy >= avg_trend


def test_breakout_atr_regime_scales_strength_and_cooldown():
    strat = BreakoutATR(timeframe="3m")
    df = _make_ohlcv(200, slope=0.6)
    df.loc[df.index[-1], "close"] += 8.0
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] + 0.3
    df.loc[df.index[-1], "volume"] = 200.0
    tf_mult = strat._tf_multiplier("3m")
    base_cooldown = strat._cooldown_for(tf_mult)
    bar = {"window": df, "timeframe": "3m", "symbol": "TREND", "volatility": 0.0}
    sig = strat.on_bar(bar)
    assert sig is not None
    assert sig.metadata["regime"] == pytest.approx(strat.last_regime)
    assert abs(sig.metadata["regime"]) >= 0.55
    assert sig.metadata["regime_threshold"] == pytest.approx(strat.last_regime_threshold)
    assert strat.min_regime <= sig.metadata["regime_threshold"] <= strat.max_regime
    assert sig.strength > 0.6
    if base_cooldown > 0:
        assert strat.cooldown_bars <= base_cooldown


def test_breakout_atr_vol_quantile_configures_threshold():
    df = _make_ohlcv(200, slope=0.5)
    df.loc[df.index[-1], "close"] += 5.0
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] + 0.3
    df.loc[df.index[-1], "volume"] = 200.0

    bar_common = {
        "window": df,
        "timeframe": "5m",
        "volatility": 0.0,
        "market_type": "perp",
    }

    strat_low = BreakoutATR(timeframe="5m", vol_quantile=0.2)
    strat_high = BreakoutATR(timeframe="5m", vol_quantile=0.4)

    bar_low = {**bar_common, "symbol": "QLOW"}
    bar_high = {**bar_common, "symbol": "QHGH"}

    strat_low.on_bar(bar_low)
    strat_high.on_bar(bar_high)

    tf_mult = strat_low._tf_multiplier("5m")
    expected_low = strat_low._vol_quantile_for(tf_mult, "perp")
    expected_high = strat_high._vol_quantile_for(tf_mult, "perp")

    rq_low = strat_low._rq._cache.get(("QLOW", "atr"))
    rq_high = strat_high._rq._cache.get(("QHGH", "atr"))

    assert rq_low is not None
    assert rq_high is not None

    assert expected_low != pytest.approx(expected_high)
    assert rq_low.q == pytest.approx(expected_low)
    assert rq_high.q == pytest.approx(expected_high)


@pytest.mark.parametrize(
    "timeframe, expected_range",
    [
        ("1m", (0.19, 0.21)),
        ("15m", (0.25, 0.4)),
        ("4h", (0.4, 0.7)),
    ],
)
def test_breakout_atr_vol_quantile_scaling(timeframe, expected_range):
    strat = BreakoutATR(vol_quantile=0.2)
    tf_mult = strat._tf_multiplier(timeframe)
    quantile = strat._vol_quantile_for(tf_mult)
    lower, upper = expected_range
    assert lower <= quantile <= upper


@pytest.mark.parametrize("timeframe", ["1m"])
def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell, timeframe, monkeypatch):
    strat = BreakoutATR(ema_n=2, atr_n=2, min_regime=0.1, max_regime=0.4, volume_factor=0.0)

    bar_common = {
        "window": breakout_df_buy,
        "volatility": 0.0,
        "timeframe": timeframe,
        "symbol": "BTC/USDT",
    }
    sig_buy = strat.on_bar(bar_common)
    tf_mult = strat._tf_multiplier(timeframe)
    ema_used = strat._ema_period(tf_mult)
    atr_used = strat._atr_period(tf_mult)
    upper, lower = keltner_channels(breakout_df_buy, ema_used, atr_used, strat.mult)
    last_close_buy = breakout_df_buy["close"].iloc[-1]
    assert sig_buy.side == "buy"
    expected_buy = float(upper.iloc[-1])
    assert sig_buy.limit_price == pytest.approx(expected_buy)
    assert sig_buy.limit_price <= last_close_buy
    assert sig_buy.metadata["post_only"] is True
    partial_tp = sig_buy.metadata.get("partial_take_profit")
    assert partial_tp is not None
    assert partial_tp["qty_pct"] > 0
    assert partial_tp["atr_multiple"] >= 1.2

    sell_df = breakout_df_sell.copy()
    sig_sell = strat.on_bar(
        {
            "window": sell_df,
            "volatility": 0.0,
            "timeframe": timeframe,
            "symbol": "BTC/USDT",
        }
    )
    upper, lower = keltner_channels(sell_df, ema_used, atr_used, strat.mult)
    last_close_sell = sell_df["close"].iloc[-1]
    assert sig_sell.side == "sell"
    expected_sell = float(lower.iloc[-1])
    assert sig_sell.limit_price == pytest.approx(expected_sell)
    assert sig_sell.limit_price >= last_close_sell
    assert sig_buy.metadata["post_only"] is True
    assert sig_sell.metadata["post_only"] is True
    assert sig_buy.post_only is True
    assert "regime" in sig_buy.metadata
    assert isinstance(sig_buy.metadata["regime"], float)
    assert sig_buy.metadata["regime_threshold"] >= strat.min_regime
    assert sig_sell.metadata["regime_threshold"] >= strat.min_regime

    # Marginal breakout should not produce strong conviction
    length = 60
    opens = [10.0 + i * 0.7 for i in range(length)]
    closes = [o + 0.4 for o in opens]
    highs = [c + 0.4 for c in closes]
    lows = [o - 0.4 for o in opens]
    volume = [1.0] * length
    closes[-1] = closes[-2] + 0.5
    highs[-1] = closes[-1] + 0.3
    marginal_df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume}
    )

    strat_marginal = BreakoutATR(
        ema_n=2,
        atr_n=2,
        min_regime=0.1,
        max_regime=0.4,
        volume_factor=0.0,
    )

    import types

    class _StaticQuantile:
        def __init__(self, name: str) -> None:
            self.name = name

        def update(self, value: float) -> float:
            if self.name == "atr_bps":
                return value / 1.02
            return value

    def _patched_get(self, symbol: str, name: str, **kwargs):
        return _StaticQuantile(name)

    monkeypatch.setattr(
        strat_marginal._rq,
        "get",
        types.MethodType(_patched_get, strat_marginal._rq),
    )

    marginal_sig = strat_marginal.on_bar(
        {
            "window": marginal_df,
            "volatility": 0.0,
            "timeframe": timeframe,
            "symbol": "BTC/USDT",
        }
    )

    assert marginal_sig is None or marginal_sig.strength <= 0.1


@pytest.mark.parametrize("timeframe", ["1m"])
def test_breakout_atr_risk_service_handles_stop_and_size(breakout_df_buy, timeframe):
    account = Account(float("inf"))
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X")
    )
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = BreakoutATR(ema_n=2, atr_n=2, **{"risk_service": svc})
    sig = strat.on_bar(
        {"window": breakout_df_buy, "volatility": 0.0, "timeframe": timeframe, "symbol": "BTC/USDT"}
    )
    assert sig and sig.side == "buy"
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    tf_mult = strat._tf_multiplier(timeframe)
    stop_mult = strat._stop_multiplier(tf_mult)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade["atr"], atr_mult=stop_mult)
    assert trade["stop"] == pytest.approx(expected_stop)
    assert trade["partial_take_profit"] == sig.metadata.get("partial_take_profit")


def test_breakout_atr_vector_signal(breakout_df_buy):
    data = breakout_df_buy.copy()
    data.loc[data.index[-1], "close"] += 10.0
    data["volume"] = 100
    entries, exits = BreakoutATR.signal(
        data, ema_n=2, atr_n=2, mult=1.0, volume_factor=0
    )
    assert entries.iloc[-1]
    assert not exits.iloc[-1]


def test_breakout_atr_limit_reprices_progressively(breakout_df_buy):
    strat = BreakoutATR(ema_n=2, atr_n=2)
    symbol = "BTC/USDT"
    sig = strat.on_bar(
        {"window": breakout_df_buy, "volatility": 0.0, "timeframe": "1m", "symbol": symbol}
    )
    assert sig and sig.side == "buy"
    last_close = float(breakout_df_buy["close"].iloc[-1])
    assert sig.limit_price <= last_close
    base_price = sig.metadata["base_price"]
    assert base_price == pytest.approx(sig.limit_price)
    initial_offset = sig.metadata["initial_offset"]
    offset_step = sig.metadata["offset_step"]
    target_offset = sig.metadata["limit_offset"]
    assert initial_offset > 0
    assert offset_step > 0
    assert target_offset >= initial_offset
    assert sig.metadata["partial_take_profit"]["qty_pct"] <= 0.6

    order = Order(symbol, sig.side, "limit", qty=1.0, price=sig.limit_price, post_only=sig.post_only)
    res = {"pending_qty": order.qty, "price": order.price}

    offsets: list[float] = []
    for _ in range(3):
        action = strat.on_order_expiry(order, res)
        assert action == "re_quote"
        offsets.append(abs(order.price - base_price))
        res["price"] = order.price

    assert offsets[0] == pytest.approx(initial_offset)
    assert offsets[1] >= offsets[0]
    assert offsets[2] >= offsets[1]
    assert offsets[-1] <= target_offset + 1e-9
    expected_step = max(0.0, min(offset_step, target_offset - offsets[0]))
    assert offsets[1] - offsets[0] == pytest.approx(expected_step, rel=1e-6, abs=1e-9)


def test_order_flow_signals():
    df_buy = pd.DataFrame(
        {
            "bid_qty": [1, 2, 3, 4, 10],
            "ask_qty": [5, 4, 3, 2, 1],
            "close": [100, 101, 100, 102, 101],
        }
    )
    df_sell = pd.DataFrame(
        {
            "bid_qty": [10, 4, 3, 2, 1],
            "ask_qty": [1, 2, 3, 4, 10],
            "close": [101, 100, 102, 99, 98],
        }
    )
    strat = OrderFlow(window=3, buy_threshold=10.0, sell_threshold=10.0)
    sig_buy = strat.on_bar({"window": df_buy, "timeframe": "1m", "symbol": "X", "volatility": 0.0})
    assert sig_buy and sig_buy.side == "buy"
    assert sig_buy.limit_price == pytest.approx(df_buy["close"].iloc[-1])
    sig_sell = strat.on_bar({"window": df_sell, "timeframe": "1m", "symbol": "X", "volatility": 0.0})
    assert sig_sell and sig_sell.side == "sell"
    assert sig_sell.limit_price == pytest.approx(df_sell["close"].iloc[-1])


def test_mean_rev_ofi_signals():
    cfg = yaml.safe_load(
        """
ofi_window: 2
zscore_threshold: 0.5
vol_window: 2
vol_threshold: 0.9
"""
    )
    closes = [100, 110, 100, 105, 100, 101]
    df_sell = pd.DataFrame(
        {
            "bid_qty": [1, 2, 3, 4, 5, 10],
            "ask_qty": [5, 4, 3, 2, 1, 0],
            "close": closes,
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_sell = strat.on_bar({"window": df_sell, "volatility": 0.0, "timeframe": "3m"})
    assert sig_sell.side == "sell"

    df_buy = pd.DataFrame(
        {
            "bid_qty": [5, 4, 3, 2, 1, 0],
            "ask_qty": [1, 2, 3, 4, 5, 10],
            "close": closes,
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_buy = strat.on_bar({"window": df_buy, "volatility": 0.0, "timeframe": "3m"})
    assert sig_buy.side == "buy"


def test_mean_rev_ofi_trailing_stop_uses_atr():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    trade = {
        "side": "sell",
        "entry_price": 100.0,
        "qty": 1.0,
        "stop": 101.0,
        "atr": 1.0,
        "stage": 3,
    }
    rm.update_trailing(trade, 90.0)
    assert trade["stop"] == pytest.approx(90.0 + 2 * trade["atr"])


def test_breakout_vol_risk_service_handles_stop_and_size():
    df_buy = pd.DataFrame(
        {
            "close": [1.0, 1.05, 1.1, 1.15, 10.0],
            "volume": [100, 105, 102, 108, 180],
        }
    )
    account = Account(float("inf"))
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X")
    )
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = BreakoutVol(lookback=2, max_offset_pct=0.0, **{"risk_service": svc})
    for end in range(2, len(df_buy)):
        strat.on_bar({"window": df_buy.iloc[: end], "volatility": 0.0})
    sig = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig and sig.side == "buy"
    assert sig.limit_price >= df_buy["close"].iloc[-1]
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade.get("atr"))
    assert trade["stop"] == pytest.approx(expected_stop)


def test_breakout_vol_more_signals_in_lower_timeframe():
    np.random.seed(0)
    prices = 100 + np.cumsum(np.random.normal(0, 0.03, 120))
    signals: dict[str, int] = {}
    for tf in ["1m", "5m"]:
        strat = BreakoutVol(lookback=10, timeframe=tf)
        sigs = 0
        for i in range(10, len(prices)):
            df = pd.DataFrame(
                {"close": prices[: i + 1], "volume": np.full(i + 1, 150.0)}
            )
            sig = strat.on_bar({"window": df})
            if sig and sig.side in {"buy", "sell"}:
                sigs += 1
        signals[tf] = sigs
    assert signals["1m"] > signals["5m"]


@settings(deadline=None)
@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_buy_property(start, inc):
    df = pd.DataFrame(
        {
            "bid_qty": [start + i * inc for i in range(3)],
            "ask_qty": [start - i * inc for i in range(3)],
        }
    )
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig is None


@settings(deadline=None)
@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_sell_property(start, inc):
    df = pd.DataFrame(
        {
            "bid_qty": [start - i * inc for i in range(3)],
            "ask_qty": [start + i * inc for i in range(3)],
        }
    )
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig is None
