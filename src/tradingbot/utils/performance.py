"""Performance metrics utilities."""
from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np


def dsr(returns: Sequence[float], num_trials: int = 1, benchmark: float = 0.0) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    Parameters
    ----------
    returns:
        Sequence of periodic returns.
    num_trials:
        Number of independent strategy variations evaluated.  Defaults to ``1``
        (no deflation).
    benchmark:
        Benchmark Sharpe ratio to beat, typically ``0``.

    Returns
    -------
    float
        Probability that the strategy's Sharpe ratio exceeds the benchmark
        after accounting for selection bias and non-normality.
    """

    r = np.asarray(list(returns), dtype=float)
    if r.size < 2:
        raise ValueError("returns must contain at least two observations")

    n = r.size
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        raise ValueError("returns must have non-zero variance")
    sr = mean / std

    m3 = np.mean((r - mean) ** 3)
    m4 = np.mean((r - mean) ** 4)
    skew = m3 / std**3
    kurt = m4 / std**4

    sr_var = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)
    if sr_var <= 0 or not np.isfinite(sr_var):
        raise ValueError("invalid return distribution")
    sr_std = np.sqrt(sr_var)

    z = NormalDist().inv_cdf(1 - 1.0 / max(num_trials, 1))
    sr_max = benchmark + sr_std * z

    return NormalDist().cdf((sr - sr_max) / sr_std)
