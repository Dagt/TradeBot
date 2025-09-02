"""Liquidity filters for strategy signals.

Provides basic checks for market liquidity conditions before allowing
strategies to emit trading signals.  The filter verifies that the current
market spread is below ``max_spread``, that traded volume exceeds
``min_volume`` and that price volatility stays below ``max_volatility``.

Thresholds are normally loaded from the application configuration when the
module is imported.  When no explicit thresholds are provided the default
filter automatically calibrates sensible values based on moving percentiles
of the most recent market data.  If a particular metric is missing from the
bar data, the check for that metric is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from collections import defaultdict, deque

from ..config.hydra_conf import load_config


@dataclass
class LiquidityFilter:
    """Simple liquidity filter with spread, volume and volatility checks."""

    max_spread: float = float("inf")
    min_volume: float = 0.0
    max_volatility: float = float("inf")

    def check(self, bar: dict[str, Any]) -> bool:
        """Return ``True`` if ``bar`` passes all liquidity checks."""

        spread = bar.get("spread")
        if spread is None and {"ask", "bid"} <= bar.keys():
            spread = float(bar["ask"]) - float(bar["bid"])
        if spread is not None and spread > self.max_spread:
            return False

        volume = bar.get("volume")
        if volume is not None and volume < self.min_volume:
            return False

        vol = bar.get("volatility")
        if vol is None:
            window = bar.get("window")
            if isinstance(window, pd.DataFrame) and "close" in window.columns and len(window) > 1:
                vol = window["close"].pct_change().dropna().std()
        if vol is not None and vol > self.max_volatility:
            return False

        return True


def _load_default_filter() -> LiquidityFilter:
    """Load thresholds from config or infer them from recent market data.

    When no thresholds are provided in ``config.yaml`` the filter estimates
    defaults from the most recent ``N`` bars using moving percentiles.  The
    95th percentile of the spread and volatility set upper bounds, while the
    5th percentile of traded volume establishes a minimum requirement.  This
    keeps behaviour transparent to the caller while still enforcing liquidity
    constraints.
    """

    cfg = load_config()
    filt_cfg = getattr(cfg, "filters", None)

    # Start with any explicitly configured thresholds
    max_spread = float(getattr(filt_cfg, "max_spread", float("inf")))
    min_volume = float(getattr(filt_cfg, "min_volume", 0.0))
    max_volatility = float(getattr(filt_cfg, "max_volatility", float("inf")))

    # If some thresholds are missing, estimate them from the latest bars
    if (
        max_spread == float("inf")
        or min_volume == 0.0
        or max_volatility == float("inf")
    ):
        bt_cfg = getattr(cfg, "backtest", None)
        data_path = getattr(bt_cfg, "data", None) if bt_cfg else None
        window = int(getattr(bt_cfg, "window", 0)) if bt_cfg else 0
        try:
            df = pd.read_csv(Path(data_path)) if data_path else pd.DataFrame()
            if window:
                df = df.tail(window)
        except Exception:  # pragma: no cover - best effort to load data
            df = pd.DataFrame()

        lookback = window or len(df)
        if max_spread == float("inf") and "spread" in df.columns and not df.empty:
            max_spread = float(df["spread"].tail(lookback).quantile(0.95))
        if min_volume == 0.0 and "volume" in df.columns and not df.empty:
            min_volume = float(df["volume"].tail(lookback).quantile(0.05))
        if max_volatility == float("inf"):
            if "volatility" in df.columns and not df.empty:
                max_volatility = float(df["volatility"].tail(lookback).quantile(0.95))
            elif {"close"} <= set(df.columns) and len(df) > 1:
                returns = df["close"].pct_change()
                vol_series = returns.rolling(lookback, min_periods=2).std().dropna()
                if not vol_series.empty:
                    max_volatility = float(vol_series.tail(lookback).quantile(0.95))

    return LiquidityFilter(
        max_spread=max_spread,
        min_volume=min_volume,
        max_volatility=max_volatility,
    )


_default_filter = _load_default_filter()

# Historical metrics per (symbol, timeframe)
_history: dict[tuple[str, str], dict[str, deque[float]]] = defaultdict(
    lambda: {"spread": deque(maxlen=1000), "volume": deque(maxlen=1000), "volatility": deque(maxlen=1000)}
)
# Cached filters per (symbol, timeframe)
_filters: dict[tuple[str, str], LiquidityFilter] = {}

# Minimum samples required before enforcing thresholds
MIN_SAMPLES = 5


def _update_metrics(symbol: str, timeframe: str, bar: dict[str, Any]) -> LiquidityFilter:
    """Record metrics for ``symbol``/``timeframe`` and recompute thresholds."""

    hist = _history[(symbol, timeframe)]

    spread = bar.get("spread")
    if spread is None and {"ask", "bid"} <= bar.keys():
        spread = float(bar["ask"]) - float(bar["bid"])
    if spread is not None:
        hist["spread"].append(float(spread))

    volume = bar.get("volume")
    if volume is not None:
        hist["volume"].append(float(volume))

    vol = bar.get("volatility")
    if vol is None:
        window = bar.get("window")
        if isinstance(window, pd.DataFrame) and "close" in window.columns and len(window) > 1:
            vol = window["close"].pct_change().dropna().std()
    if vol is not None:
        hist["volatility"].append(float(vol))

    filt = _filters.get((symbol, timeframe), LiquidityFilter())
    if hist["spread"]:
        filt.max_spread = float(pd.Series(hist["spread"]).quantile(0.95))
    if hist["volume"]:
        filt.min_volume = float(pd.Series(hist["volume"]).quantile(0.05))
    if hist["volatility"]:
        filt.max_volatility = float(pd.Series(hist["volatility"]).quantile(0.95))

    _filters[(symbol, timeframe)] = filt
    return filt


def passes(
    bar: dict[str, Any],
    timeframe: str | None = None,
    filt: LiquidityFilter | None = None,
) -> bool:
    """Return ``True`` if ``bar`` passes liquidity checks.

    Parameters
    ----------
    bar:
        Market data bar containing metrics like ``bid``, ``ask``, ``volume``
        or a pandas ``DataFrame`` under the ``window`` key.
    timeframe:
        Optional timeframe string.  When provided along with a ``symbol`` in
        ``bar``, historical metrics are tracked separately for each
        ``(symbol, timeframe)`` combination.
    filt:
        Optional :class:`LiquidityFilter` instance.  If provided it takes
        precedence over the cached filters.
    """

    if filt is not None:
        return filt.check(bar)

    symbol = bar.get("symbol")
    tf = timeframe or bar.get("timeframe")
    if symbol and tf:
        hist = _history[(symbol, tf)]
        filt = _filters.get((symbol, tf), LiquidityFilter())
        samples = max(len(hist["spread"]), len(hist["volume"]), len(hist["volatility"]))
        if samples >= MIN_SAMPLES and not filt.check(bar):
            return False
        _update_metrics(symbol, tf, bar)
        return True
    else:
        return _default_filter.check(bar)
