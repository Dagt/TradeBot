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
    assert abs(strat.last_regime) < 0.3


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
    assert sig.strength > 0.6
    if base_cooldown > 0:
        assert strat.cooldown_bars <= base_cooldown


@pytest.mark.parametrize("timeframe", ["1m"])
def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell, timeframe):
    strat = BreakoutATR(ema_n=2, atr_n=2)

    bar_common = {
        "window": breakout_df_buy,
        "volatility": 0.0,
        "timeframe": timeframe,
        "symbol": "BTC/USDT",
    }
    sig_buy = strat.on_bar(bar_common)
    tf_mult = strat._tf_multiplier(timeframe)
    offset_frac = strat._offset_fraction(tf_mult)
    ema_used = strat._ema_period(tf_mult)
    atr_used = strat._atr_period(tf_mult)
    atr_buy = atr(breakout_df_buy, atr_used).dropna().iloc[-1]
    upper, lower = keltner_channels(breakout_df_buy, ema_used, atr_used, strat.mult)
    last_close_buy = breakout_df_buy["close"].iloc[-1]
    assert sig_buy.side == "buy"
    abs_price_buy = abs(last_close_buy)
    offset_buy = float(atr_buy) * offset_frac
    max_offset_buy = abs_price_buy * 0.006
    min_offset_buy = abs_price_buy * 0.0005
    offset_buy = max(min_offset_buy, min(offset_buy, max_offset_buy))
    expected_buy = max(
        last_close_buy,
        float(upper.iloc[-1]),
    )
    expected_buy += offset_buy
    assert sig_buy.limit_price == pytest.approx(expected_buy)
    assert sig_buy.limit_price >= last_close_buy

    sell_df = breakout_df_sell.copy()
    sig_sell = strat.on_bar(
        {
            "window": sell_df,
            "volatility": 0.0,
            "timeframe": timeframe,
            "symbol": "BTC/USDT",
        }
    )
    atr_sell = atr(sell_df, atr_used).dropna().iloc[-1]
    upper, lower = keltner_channels(sell_df, ema_used, atr_used, strat.mult)
    last_close_sell = sell_df["close"].iloc[-1]
    assert sig_sell.side == "sell"
    abs_price_sell = abs(last_close_sell)
    offset_sell = float(atr_sell) * offset_frac
    max_offset_sell = abs_price_sell * 0.006
    min_offset_sell = abs_price_sell * 0.0005
    offset_sell = max(min_offset_sell, min(offset_sell, max_offset_sell))
    expected_sell = min(
        last_close_sell,
        float(lower.iloc[-1]),
    )
    expected_sell -= offset_sell
    assert sig_sell.limit_price == pytest.approx(expected_sell)
    assert sig_sell.limit_price <= last_close_sell
    assert "regime" in sig_buy.metadata
    assert isinstance(sig_buy.metadata["regime"], float)


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


def test_breakout_atr_vector_signal(breakout_df_buy):
    data = breakout_df_buy.copy()
    data.loc[data.index[-1], "close"] += 10.0
    data["volume"] = 100
    entries, exits = BreakoutATR.signal(
        data, ema_n=2, atr_n=2, mult=1.0, volume_factor=0
    )
    assert entries.iloc[-1]
    assert not exits.iloc[-1]


@pytest.mark.asyncio
async def test_breakout_atr_limit_crosses_market_gets_fill(
    breakout_df_buy, paper_adapter
):
    strat = BreakoutATR(ema_n=2, atr_n=2)
    symbol = "BTC/USDT"
    sig = strat.on_bar(
        {"window": breakout_df_buy, "volatility": 0.0, "timeframe": "1m", "symbol": symbol}
    )
    assert sig and sig.side == "buy"

    tf_mult = strat._tf_multiplier("1m")
    ema_used = strat._ema_period(tf_mult)
    atr_used = strat._atr_period(tf_mult)
    atr_buy = atr(breakout_df_buy, atr_used).dropna().iloc[-1]
    upper, _ = keltner_channels(breakout_df_buy, ema_used, atr_used, strat.mult)
    last_close = float(breakout_df_buy["close"].iloc[-1])
    offset = float(atr_buy) * strat._offset_fraction(tf_mult)
    max_offset = last_close * 0.006
    min_offset = last_close * 0.0005
    offset = max(min_offset, min(offset, max_offset))
    expected_limit = max(last_close, float(upper.iloc[-1])) + offset
    assert sig.limit_price == pytest.approx(expected_limit)
    assert sig.limit_price >= last_close

    paper_adapter.update_last_price(symbol, last_close)
    res = await paper_adapter.place_order(
        symbol,
        sig.side,
        "limit",
        qty=1.0,
        price=sig.limit_price,
    )
    assert res["status"] == "filled"
    assert res["price"] == pytest.approx(last_close)


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
    df_buy = pd.DataFrame({"close": [1, 2, 3, 10]})
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
    strat = BreakoutVol(lookback=2, **{"risk_service": svc})
    sig = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig and sig.side == "buy"
    assert sig.limit_price == pytest.approx(df_buy["close"].iloc[-1])
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
            df = pd.DataFrame({"close": prices[: i + 1]})
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
