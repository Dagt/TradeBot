"""Tests for dynamic liquidity filter thresholds."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.filters import liquidity


def test_default_filter_estimates_thresholds(tmp_path, monkeypatch):
    """When no thresholds are provided the filter adapts to recent data."""

    df = pd.DataFrame(
        {
            "spread": [1.0, 2.0, 1.5, 2.5, 2.0],
            "volume": [100, 150, 200, 120, 180],
            "volatility": [0.01, 0.02, 0.015, 0.03, 0.025],
        }
    )
    csv_path = tmp_path / "bars.csv"
    df.to_csv(csv_path, index=False)

    cfg = SimpleNamespace(
        filters=SimpleNamespace(
            max_spread=float("inf"),
            min_volume=0.0,
            max_volatility=float("inf"),
            volume_quantile=0.5,
            spread_quantile=0.5,
        ),
        backtest=SimpleNamespace(data=str(csv_path), window=5),
    )

    monkeypatch.setattr("tradingbot.config.hydra_conf.load_config", lambda path=None: cfg)
    importlib.reload(liquidity)
    try:
        filt = liquidity._default_filter
        assert filt.max_spread == pytest.approx(df["spread"].quantile(0.5))
        assert filt.min_volume == pytest.approx(df["volume"].quantile(0.5))
        assert filt.max_volatility == pytest.approx(2 * df["volatility"].median())
    finally:
        monkeypatch.undo()
        importlib.reload(liquidity)

