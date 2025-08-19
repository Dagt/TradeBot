import pandas as pd
import numpy as np
import pytest

from tradingbot.data.features import (
    calc_ofi,
    order_flow_imbalance,
    depth_imbalance,
    rsi,
    atr,
    atr_ewma,
    returns,
    donchian_channels,
)


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def orderbook_df():
    return pd.DataFrame({
        "bid_qty": [5, 7, 3, 4],
        "ask_qty": [8, 6, 5, 7],
    })


@pytest.fixture
def orderbook_snapshots(orderbook_df):
    """Synthetic L2 snapshots with quantities stored as arrays."""

    snaps = []
    for _, row in orderbook_df.iterrows():
        snaps.append(
            {
                "bid_qty": [row.bid_qty, row.bid_qty - 1],
                "ask_qty": [row.ask_qty, row.ask_qty + 1],
            }
        )
    return snaps


@pytest.fixture
def l2_snapshots():
    """Synthetic L2 snapshots with price and quantity depth."""

    return [
        {
            "bid_px": [100.0, 99.0],
            "bid_qty": [5.0, 3.0],
            "ask_px": [101.0, 102.0],
            "ask_qty": [8.0, 2.0],
        },
        {
            "bid_px": [100.5, 100.0],
            "bid_qty": [6.0, 1.0],
            "ask_px": [101.5, 102.0],
            "ask_qty": [5.0, 4.0],
        },
        {
            "bid_px": [100.5, 99.5],
            "bid_qty": [4.0, 2.0],
            "ask_px": [101.5, 102.5],
            "ask_qty": [6.0, 3.0],
        },
    ]


@pytest.fixture
def close_df():
    return pd.DataFrame({"close": list(range(1, 30))})


@pytest.fixture
def close_snapshots(close_df):
    return close_df.to_dict(orient="records")


@pytest.fixture
def price_df():
    return pd.DataFrame({"close": [100, 101, 102, 101]})


@pytest.fixture
def price_snapshots(price_df):
    return price_df.to_dict(orient="records")


@pytest.fixture
def ohlc_df():
    return pd.DataFrame(
        {
            "high": [10, 11, 14, 13, 15],
            "low": [7, 9, 10, 8, 12],
            "close": [9, 10, 12, 11, 14],
        }
    )


@pytest.fixture
def ohlc_snapshots(ohlc_df):
    return ohlc_df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Order book features


@pytest.mark.parametrize("func", [order_flow_imbalance, calc_ofi])
def test_order_flow_imbalance_df(orderbook_df, func):
    ofi = func(orderbook_df)
    assert list(ofi) == [0, 4, -3, -1]


@pytest.mark.parametrize("func", [order_flow_imbalance, calc_ofi])
def test_order_flow_imbalance_snapshots(orderbook_snapshots, func):
    ofi = func(orderbook_snapshots)
    assert list(ofi) == [0, 4, -3, -1]


def test_depth_imbalance_df(orderbook_df):
    di = depth_imbalance(orderbook_df)
    expected = [
        (5 - 8) / (5 + 8),
        (7 - 6) / (7 + 6),
        (3 - 5) / (3 + 5),
        (4 - 7) / (4 + 7),
    ]
    assert all(abs(a - b) < 1e-9 for a, b in zip(di, expected))


def test_depth_imbalance_snapshots(orderbook_snapshots):
    di = depth_imbalance(orderbook_snapshots)
    expected = [
        (5 - 8) / (5 + 8),
        (7 - 6) / (7 + 6),
        (3 - 5) / (3 + 5),
        (4 - 7) / (4 + 7),
    ]
    assert all(abs(a - b) < 1e-9 for a, b in zip(di, expected))


@pytest.mark.parametrize("func", [order_flow_imbalance, calc_ofi])
def test_order_flow_imbalance_l2(l2_snapshots, func):
    ofi = func(l2_snapshots, depth=2)
    assert list(ofi) == [0.0, 13.0, 0.0]


def test_depth_imbalance_l2(l2_snapshots):
    di = depth_imbalance(l2_snapshots, depth=2)
    expected = [
        (5 + 3 - (8 + 2)) / (5 + 3 + 8 + 2),
        (6 + 1 - (5 + 4)) / (6 + 1 + 5 + 4),
        (4 + 2 - (6 + 3)) / (4 + 2 + 6 + 3),
    ]
    assert all(abs(a - b) < 1e-9 for a, b in zip(di, expected))


# ---------------------------------------------------------------------------
# Momentum / volatility indicators


def test_rsi_range(close_df):
    values = rsi(close_df, 14)
    assert values.iloc[-1] > 70  # strong uptrend
    assert values.dropna().between(0, 100).all()


def test_rsi_snapshots(close_df, close_snapshots):
    rsi_df = rsi(close_df, 14)
    rsi_snaps = rsi(close_snapshots, 14)
    pd.testing.assert_series_equal(rsi_df, rsi_snaps)


def test_atr_df(ohlc_df):
    # Last value should be the average of the last three true ranges
    val = atr(ohlc_df, n=3).iloc[-1]
    assert abs(val - 4.333333333333333) < 1e-9


def test_atr_snapshots(ohlc_df, ohlc_snapshots):
    atr_df = atr(ohlc_df, n=3)
    atr_snaps = atr(ohlc_snapshots, n=3)
    pd.testing.assert_series_equal(atr_df, atr_snaps)


def test_atr_ewma_df(ohlc_df):
    # EWMA variant produces a smoother estimate of true range
    val = atr_ewma(ohlc_df, n=3).iloc[-1]
    assert abs(val - 3.6296296296296293) < 1e-9


def test_atr_ewma_snapshots(ohlc_df, ohlc_snapshots):
    atr_df = atr_ewma(ohlc_df, n=3)
    atr_snaps = atr_ewma(ohlc_snapshots, n=3)
    pd.testing.assert_series_equal(atr_df, atr_snaps)


def test_returns_df(price_df):
    ret = returns(price_df)
    expected = np.log(price_df.close / price_df.close.shift(1))
    expected.name = "returns"
    pd.testing.assert_series_equal(ret, expected)


def test_returns_snapshots(price_df, price_snapshots):
    ret_df = returns(price_df)
    ret_snaps = returns(price_snapshots)
    pd.testing.assert_series_equal(ret_df, ret_snaps)


def test_donchian_channels_df(ohlc_df):
    upper, lower = donchian_channels(ohlc_df, n=3)
    expected_upper = pd.Series([np.nan, np.nan, 14, 14, 15], name="high")
    expected_lower = pd.Series([np.nan, np.nan, 7, 8, 8], name="low")
    pd.testing.assert_series_equal(upper, expected_upper)
    pd.testing.assert_series_equal(lower, expected_lower)


def test_donchian_channels_snapshots(ohlc_df, ohlc_snapshots):
    u_df, l_df = donchian_channels(ohlc_df, n=3)
    u_snaps, l_snaps = donchian_channels(ohlc_snapshots, n=3)
    pd.testing.assert_series_equal(u_df, u_snaps)
    pd.testing.assert_series_equal(l_df, l_snaps)


