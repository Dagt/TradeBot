"""Unit tests for synthetic spread fallback in :class:`SlippageModel`."""

from __future__ import annotations

import pytest

from tradingbot.backtesting.engine import SlippageModel


def test_compute_spread_uses_ohlc_range_without_bba() -> None:
    model = SlippageModel(
        source="bba",
        base_spread=0.1,
        ohlc_spread_factor=0.25,
        pct=0.0,
        spread_mult=1.0,
    )
    bar = {"high": 105.0, "low": 95.0, "close": 100.0}

    spread = model._compute_spread(bar)

    expected = max(0.1, (105.0 - 95.0) * 0.25)
    assert spread == pytest.approx(expected)
    assert spread > 0.0


def test_compute_spread_uses_close_bps_without_bba() -> None:
    model = SlippageModel(
        source="bba",
        base_spread=0.0,
        ohlc_spread_bps=25.0,
        pct=0.0,
        spread_mult=1.0,
    )
    bar = {"close": 200.0}

    spread = model._compute_spread(bar)

    expected = 200.0 * 25.0 / 10000.0
    assert spread == pytest.approx(expected)
    assert spread > 0.0


def test_compute_spread_combines_range_and_bps_without_bba() -> None:
    model = SlippageModel(
        source="bba",
        base_spread=0.05,
        ohlc_spread_factor=0.1,
        ohlc_spread_bps=10.0,
        pct=0.0,
        spread_mult=1.0,
    )
    bar = {"high": 101.0, "low": 99.5, "close": 100.0}

    spread = model._compute_spread(bar)

    range_component = (101.0 - 99.5) * 0.1
    bps_component = 100.0 * 10.0 / 10000.0
    expected = max(0.05, range_component, bps_component)
    assert spread == pytest.approx(expected)
    assert spread > 0.0
