import pandas as pd
import pytest

from tradingbot.data.features import (
    order_flow_imbalance,
    depth_imbalance,
    rsi,
    atr,
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
def close_df():
    return pd.DataFrame({"close": list(range(1, 30))})


@pytest.fixture
def close_snapshots(close_df):
    return close_df.to_dict(orient="records")


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


def test_order_flow_imbalance_df(orderbook_df):
    ofi = order_flow_imbalance(orderbook_df)
    assert list(ofi) == [0, 4, -3, -1]


def test_order_flow_imbalance_snapshots(orderbook_snapshots):
    ofi = order_flow_imbalance(orderbook_snapshots)
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


