import pandas as pd
import yaml
import pytest
from tradingbot.data.features import keltner_channels
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.order_flow import OrderFlow
from tradingbot.strategies.mean_rev_ofi import MeanRevOFI
from tradingbot.strategies.breakout_vol import BreakoutVol
from hypothesis import given, strategies as st


def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0)

    sig_buy = strat.on_bar({"window": breakout_df_buy, "volatility": 0.0})
    assert sig_buy.side == "buy"
    upper, _ = keltner_channels(breakout_df_buy, 2, 2, 1.0)
    last_close = breakout_df_buy["close"].iloc[-1]
    expected_edge = (last_close - upper.iloc[-1]) / abs(last_close) * 10000
    assert sig_buy.expected_edge_bps == pytest.approx(expected_edge)

    # ensure at least one bar passes before opposite signal
    row = {
        "open": 4,
        "high": 5,
        "low": 3.5,
        "close": -20.0,
        "volume": 1,
    }
    df_wait = pd.concat(
        [breakout_df_sell, pd.DataFrame([row])],
        ignore_index=True,
    )

    sig_sell = strat.on_bar({"window": df_wait, "volatility": 0.0})
    assert sig_sell.side == "sell"


def test_breakout_atr_min_edge(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=200)
    assert strat.on_bar({"window": breakout_df_buy, "volatility": 0.0}) is None
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=100)
    assert (
        strat.on_bar({"window": breakout_df_buy, "volatility": 0.0}).side == "buy"
    )
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=1000)
    assert strat.on_bar({"window": breakout_df_sell, "volatility": 0.0}) is None
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=900)
    assert (
        strat.on_bar({"window": breakout_df_sell, "volatility": 0.0}).side
        == "sell"
    )


def test_breakout_atr_min_volatility(breakout_df_buy):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_volatility=2000)
    assert strat.on_bar({"window": breakout_df_buy, "volatility": 0.0}) is None


def test_order_flow_signals():
    df_buy = pd.DataFrame({
        "bid_qty": [1, 2, 3, 5],
        "ask_qty": [5, 4, 3, 2],
    })
    df_sell = pd.DataFrame({
        "bid_qty": [5, 4, 3, 2],
        "ask_qty": [1, 2, 3, 5],
    })
    strat = OrderFlow(window=3, buy_threshold=1.0, sell_threshold=1.0)
    sig_buy = strat.on_bar({"window": df_buy})
    assert sig_buy.side == "buy"
    sig_sell = strat.on_bar({"window": df_sell})
    assert sig_sell.side == "sell"


def test_mean_rev_ofi_signals():
    cfg = yaml.safe_load(
        """
ofi_window: 2
zscore_threshold: 0.5
vol_window: 2
vol_threshold: 1.0
"""
    )
    df_sell = pd.DataFrame(
        {
            "bid_qty": [1, 2, 10],
            "ask_qty": [3, 1, 0],
            "close": [100, 100, 100],
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_sell = strat.on_bar({"window": df_sell, "volatility": 0.0})
    assert sig_sell.side == "sell"

    df_buy = pd.DataFrame(
        {
            "bid_qty": [3, 1, 0],
            "ask_qty": [1, 2, 10],
            "close": [100, 100, 100],
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_buy = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig_buy.side == "buy"


def test_breakout_vol_min_edge():
    df_buy = pd.DataFrame({"close": [1, 2, 3, 10]})
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=1100)
    assert strat.on_bar({"window": df_buy, "volatility": 0.0}) is None
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=1000)
    sig_buy = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig_buy.side == "buy"
    mean = df_buy["close"].rolling(2).mean().iloc[-1]
    std = df_buy["close"].rolling(2).std().iloc[-1]
    expected_edge = (df_buy["close"].iloc[-1] - (mean + 0.5 * std)) / abs(df_buy["close"].iloc[-1]) * 10000
    assert sig_buy.expected_edge_bps == pytest.approx(expected_edge)
    df_sell = pd.DataFrame({"close": [1, 2, 3, -5]})
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=2400)
    assert strat.on_bar({"window": df_sell, "volatility": 0.0}) is None
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=2000)
    sig_sell = strat.on_bar({"window": df_sell, "volatility": 0.0})
    assert sig_sell.side == "sell"
    mean = df_sell["close"].rolling(2).mean().iloc[-1]
    std = df_sell["close"].rolling(2).std().iloc[-1]
    expected_edge = ((mean - 0.5 * std) - df_sell["close"].iloc[-1]) / abs(df_sell["close"].iloc[-1]) * 10000
    assert sig_sell.expected_edge_bps == pytest.approx(expected_edge)


def test_breakout_vol_min_volatility():
    df_buy = pd.DataFrame({"close": [1, 2, 3, 10]})
    strat = BreakoutVol(lookback=2, mult=0.5, min_volatility=20000)
    assert strat.on_bar({"window": df_buy, "volatility": 0.0}) is None


@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_buy_property(start, inc):
    df = pd.DataFrame({
        "bid_qty": [start + i * inc for i in range(3)],
        "ask_qty": [start - i * inc for i in range(3)],
    })
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig.side == "buy"


@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_sell_property(start, inc):
    df = pd.DataFrame({
        "bid_qty": [start - i * inc for i in range(3)],
        "ask_qty": [start + i * inc for i in range(3)],
    })
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig.side == "sell"
