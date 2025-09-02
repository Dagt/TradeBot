"""Tests for dynamic liquidity filter thresholds."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.filters import liquidity


@pytest.mark.parametrize(
    "spreads, volumes, vols",
    [
        (
            [1.0, 2.0, 1.5, 2.5, 2.0],
            [100, 150, 200, 120, 180],
            [0.01, 0.02, 0.015, 0.03, 0.025],
        ),
        (
            [0.3, 0.4, 0.35, 0.5, 0.45],
            [500, 550, 520, 580, 560],
            [0.005, 0.006, 0.0055, 0.007, 0.0065],
        ),
    ],
    ids=["BTC-1m", "ETH-5m"],
)

def test_default_filter_estimates_percentiles(tmp_path, monkeypatch, spreads, volumes, vols):
    """Filter derives thresholds from moving percentiles when unset."""

    df = pd.DataFrame({"spread": spreads, "volume": volumes, "volatility": vols})
    csv_path = tmp_path / "bars.csv"
    df.to_csv(csv_path, index=False)

    cfg = SimpleNamespace(
        filters=SimpleNamespace(
            max_spread=float("inf"),
            min_volume=0.0,
            max_volatility=float("inf"),
        ),
        backtest=SimpleNamespace(data=str(csv_path), window=len(df)),
    )

    monkeypatch.setattr("tradingbot.config.hydra_conf.load_config", lambda path=None: cfg)
    importlib.reload(liquidity)
    try:
        filt = liquidity._default_filter
        assert filt.max_spread == pytest.approx(df["spread"].quantile(0.95))
        assert filt.min_volume == pytest.approx(df["volume"].quantile(0.05))
        assert filt.max_volatility == pytest.approx(df["volatility"].quantile(0.95))
    finally:
        monkeypatch.undo()
        importlib.reload(liquidity)


def test_filter_thresholds_vary_by_dataset(tmp_path, monkeypatch):
    """Different datasets yield different automatic thresholds."""

    df1 = pd.DataFrame(
        {
            "spread": [1.0, 2.0, 1.5, 2.5, 2.0],
            "volume": [100, 150, 200, 120, 180],
            "volatility": [0.01, 0.02, 0.015, 0.03, 0.025],
        }
    )
    df2 = pd.DataFrame(
        {
            "spread": [0.3, 0.4, 0.35, 0.5, 0.45],
            "volume": [500, 550, 520, 580, 560],
            "volatility": [0.005, 0.006, 0.0055, 0.007, 0.0065],
        }
    )

    csv1 = tmp_path / "btc.csv"
    csv2 = tmp_path / "eth.csv"
    df1.to_csv(csv1, index=False)
    df2.to_csv(csv2, index=False)

    def load(path):
        return SimpleNamespace(
            filters=SimpleNamespace(
                max_spread=float("inf"),
                min_volume=0.0,
                max_volatility=float("inf"),
            ),
            backtest=SimpleNamespace(data=str(path), window=5),
        )

    monkeypatch.setattr("tradingbot.config.hydra_conf.load_config", lambda path=None: load(csv1))
    importlib.reload(liquidity)
    filt1 = liquidity._default_filter

    monkeypatch.setattr("tradingbot.config.hydra_conf.load_config", lambda path=None: load(csv2))
    importlib.reload(liquidity)
    filt2 = liquidity._default_filter

    assert filt1.max_spread != filt2.max_spread
    assert filt1.min_volume != filt2.min_volume
    assert filt1.max_volatility != filt2.max_volatility

    monkeypatch.undo()
    importlib.reload(liquidity)


def test_thresholds_adjust_per_timeframe():
    """Each timeframe maintains its own dynamic thresholds."""

    importlib.reload(liquidity)
    symbol = "BTC"

    bars_1h = [
        {"symbol": symbol, "timeframe": "1h", "spread": s, "volume": v, "volatility": vol}
        for s, v, vol in zip(
            [10, 12, 11, 9, 13],
            [1000, 1100, 1200, 900, 1050],
            [0.10, 0.12, 0.11, 0.09, 0.13],
        )
    ]
    for bar in bars_1h:
        assert liquidity.passes(bar, bar["timeframe"])  # populate history

    filt_1h = liquidity._filters[(symbol, "1h")]

    bars_1m = [
        {"symbol": symbol, "timeframe": "1m", "spread": s, "volume": v, "volatility": vol}
        for s, v, vol in zip(
            [1.0, 1.2, 1.1, 0.9, 1.3],
            [100, 110, 90, 105, 95],
            [0.01, 0.012, 0.011, 0.009, 0.013],
        )
    ]
    for bar in bars_1m:
        assert liquidity.passes(bar, bar["timeframe"])  # populate history

    filt_1m = liquidity._filters[(symbol, "1m")]

    assert filt_1h.max_spread != filt_1m.max_spread
    assert filt_1h.min_volume != filt_1m.min_volume
    assert filt_1h.max_volatility != filt_1m.max_volatility

    # verify that a wide 1m spread fails while similar 1h spread still passes
    wide_1m = {"symbol": symbol, "timeframe": "1m", "spread": filt_1m.max_spread * 1.5, "volume": 100, "volatility": 0.01}
    assert not liquidity.passes(wide_1m, wide_1m["timeframe"])

    ok_1h = {"symbol": symbol, "timeframe": "1h", "spread": filt_1h.max_spread * 0.9, "volume": 1000, "volatility": 0.1}
    assert liquidity.passes(ok_1h, ok_1h["timeframe"])  # should still pass under 1h thresholds
