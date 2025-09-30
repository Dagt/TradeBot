import math

import pytest

from tradingbot.backtesting.engine import SlippageModel


def test_compute_spread_uses_bar_range_when_bba_missing():
    bar = {"high": 105.0, "low": 95.0, "close": 100.0}
    model = SlippageModel(
        source="bba",
        base_spread=0.1,
        spread_mult=1.0,
        ohlc_spread_factor=1.0,
        spread_bps=0.0,
    )

    spread = model._compute_spread(bar)

    expected = max(0.1, (105.0 - 95.0) * 1.0)
    assert spread == pytest.approx(expected)
    assert spread > 0.0


def test_compute_spread_uses_close_bps_when_range_missing():
    bar = {"high": math.nan, "low": math.nan, "close": 200.0}
    model = SlippageModel(
        source="bba",
        base_spread=0.0,
        spread_mult=1.0,
        ohlc_spread_factor=0.0,
        spread_bps=25.0,
    )

    spread = model._compute_spread(bar)

    expected = 200.0 * (25.0 / 10000.0)
    assert spread == pytest.approx(expected)
    assert spread > 0.0
