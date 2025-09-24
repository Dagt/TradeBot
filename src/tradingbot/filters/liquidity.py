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
from threading import Lock

import pandas as pd
from collections import Counter, defaultdict, deque

from ..config.hydra_conf import load_config


@dataclass
class LiquidityFilter:
    """Simple liquidity filter with spread, volume and volatility checks."""

    max_spread: float = float("inf")
    min_volume: float = 0.0
    max_volatility: float = float("inf")

    def check_with_reason(self, bar: dict[str, Any]) -> tuple[bool, str | None]:
        """Return pass/fail flag and the violated metric if any."""

        spread = bar.get("spread")
        if spread is None and {"ask", "bid"} <= bar.keys():
            spread = float(bar["ask"]) - float(bar["bid"])
        if spread is not None and spread > self.max_spread:
            return False, "spread"

        volume = bar.get("volume")
        if volume is not None and volume < self.min_volume:
            return False, "volume"

        vol = bar.get("volatility")
        if vol is None:
            window = bar.get("window")
            if isinstance(window, pd.DataFrame) and "close" in window.columns and len(window) > 1:
                vol = window["close"].pct_change().dropna().std()
        if vol is not None and vol > self.max_volatility:
            return False, "volatility"

        return True, None

    def check(self, bar: dict[str, Any]) -> bool:
        ok, _ = self.check_with_reason(bar)
        return ok


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


class LiquidityFilterManager:
    """Manage liquidity filters and historical metrics with thread safety."""

    MIN_SAMPLES = 5

    def __init__(self) -> None:
        self._lock = Lock()
        self._default_filter = _load_default_filter()
        self._history: dict[tuple[str, str], dict[str, deque[float]]] = defaultdict(
            lambda: {
                "spread": deque(maxlen=1000),
                "volume": deque(maxlen=1000),
                "volatility": deque(maxlen=1000),
            }
        )
        self._filters: dict[tuple[str, str], LiquidityFilter] = {}
        self._stats: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def _normalize_timeframe(self, timeframe: str | int | float | None) -> str:
        if timeframe is None:
            return "1m"
        if isinstance(timeframe, (int, float)):
            if timeframe <= 0:
                return "1m"
            return f"{float(timeframe)}m" if timeframe < 1 else f"{int(timeframe)}m"
        return str(timeframe).lower() or "1m"

    def _update_metrics(self, symbol: str, timeframe: str, bar: dict[str, Any]) -> LiquidityFilter:
        """Record metrics for ``symbol``/``timeframe`` and recompute thresholds."""

        tf_norm = self._normalize_timeframe(timeframe)
        hist = self._history[(symbol, tf_norm)]

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

        filt = self._filters.get((symbol, tf_norm), LiquidityFilter())
        if hist["spread"]:
            series = pd.Series(hist["spread"])
            median = float(series.median())
            p90 = float(series.quantile(0.9))
            p95 = float(series.quantile(0.95))
            limit = max(p95, p90 * 1.2, median * 3.0)
            filt.max_spread = max(limit, median * 1.2)
        if hist["volume"]:
            series = pd.Series(hist["volume"])
            median = float(series.median())
            p10 = float(series.quantile(0.1))
            p05 = float(series.quantile(0.05))
            floor = min(p10, p05) * 0.5
            adaptive = max(floor, median * 0.1)
            filt.min_volume = max(0.0, adaptive)
        if hist["volatility"]:
            series = pd.Series(hist["volatility"])
            median = float(series.median())
            p95 = float(series.quantile(0.95))
            p90 = float(series.quantile(0.9))
            cap = max(p95 * 1.5, p90 * 2.0, median * 5.0)
            filt.max_volatility = max(cap, median * 2.0)

        self._filters[(symbol, tf_norm)] = filt
        return filt

    def passes(
        self,
        bar: dict[str, Any],
        timeframe: str | None = None,
        filt: LiquidityFilter | None = None,
    ) -> bool:
        """Return ``True`` if ``bar`` passes liquidity checks."""

        if filt is not None:
            return filt.check(bar)

        symbol = bar.get("symbol")
        tf = timeframe or bar.get("timeframe") or getattr(self, "default_timeframe", "1m")
        tf_norm = self._normalize_timeframe(tf)
        if symbol and tf:
            with self._lock:
                hist = self._history[(symbol, tf_norm)]
                filt = self._filters.get((symbol, tf_norm), LiquidityFilter())
                samples = max(
                    len(hist["spread"]),
                    len(hist["volume"]),
                    len(hist["volatility"]),
                )
                passed, reason = filt.check_with_reason(bar)
                if samples >= self.MIN_SAMPLES and not passed:
                    info = bar.setdefault("_liquidity", {})
                    info["reason"] = reason or "unknown"
                    info["thresholds"] = {
                        "max_spread": filt.max_spread,
                        "min_volume": filt.min_volume,
                        "max_volatility": filt.max_volatility,
                    }
                    info["timeframe"] = tf_norm
                    self._stats[(symbol, tf_norm)][info["reason"]] += 1
                    self._update_metrics(symbol, tf_norm, bar)
                    return False
                self._update_metrics(symbol, tf_norm, bar)
            return True
        else:
            passed, reason = self._default_filter.check_with_reason(bar)
            if not passed:
                info = bar.setdefault("_liquidity", {})
                info["reason"] = reason or "unknown"
            return passed


__all__ = ["LiquidityFilter", "LiquidityFilterManager"]
