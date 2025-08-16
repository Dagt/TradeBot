import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .base import Strategy, Signal, record_signal_metrics


def triple_barrier_labels(
    prices: pd.Series,
    horizon: int = 5,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
) -> pd.Series:
    """Generate triple-barrier labels for a price series.

    Parameters
    ----------
    prices: pd.Series
        Series of prices.
    horizon: int, optional
        Number of future bars to inspect, by default ``5``.
    upper_pct: float, optional
        Upper barrier percentage, by default ``0.02``.
    lower_pct: float, optional
        Lower barrier percentage, by default ``0.02``.

    Returns
    -------
    pd.Series
        Labels ``1`` if the upper barrier is hit first, ``-1`` for the
        lower barrier, or ``0`` if neither is reached within the horizon.
    """

    labels = pd.Series(0, index=prices.index, dtype=int)
    n = len(prices)
    for i in range(n - 1):
        start = prices.iloc[i]
        upper = start * (1 + upper_pct)
        lower = start * (1 - lower_pct)
        end = min(i + 1 + horizon, n)
        future = prices.iloc[i + 1:end]
        hit_upper = future[future >= upper]
        hit_lower = future[future <= lower]
        if not hit_upper.empty and not hit_lower.empty:
            first_up = hit_upper.index[0]
            first_down = hit_lower.index[0]
            labels.iloc[i] = 1 if first_up < first_down else -1
        elif not hit_upper.empty:
            labels.iloc[i] = 1
        elif not hit_lower.empty:
            labels.iloc[i] = -1
    return labels


class TripleBarrier(Strategy):
    """Strategy using triple-barrier labeling and a gradient boosting model."""

    name = "triple_barrier"

    def __init__(
        self,
        horizon: int = 5,
        upper_pct: float = 0.02,
        lower_pct: float = 0.02,
        training_window: int = 200,
    ) -> None:
        self.horizon = int(horizon)
        self.upper_pct = float(upper_pct)
        self.lower_pct = float(lower_pct)
        self.training_window = int(training_window)
        self.model = GradientBoostingClassifier()
        self.fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["close"].pct_change().fillna(0)
        feat = pd.DataFrame(
            {
                "ret": returns,
                "ret_mean": returns.rolling(5).mean().fillna(0),
                "ret_std": returns.rolling(5).std().fillna(0),
            }
        )
        return feat

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.training_window:
            return None
        features = self._prepare_features(df)
        if not self.fitted:
            labels = triple_barrier_labels(
                df["close"], self.horizon, self.upper_pct, self.lower_pct
            )
            X = features.iloc[:-1]
            y = labels.iloc[:-1]
            if y.nunique() > 1:
                self.model.fit(X, y)
                self.fitted = True
            else:
                return None
        x_last = features.iloc[[-1]]
        pred = self.model.predict(x_last)[0]
        if pred == 1:
            return Signal("buy", 1.0)
        if pred == -1:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
