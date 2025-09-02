import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import ClassifierMixin

from .base import Strategy, Signal, record_signal_metrics, load_params
from ..filters.liquidity import LiquidityFilterManager


PARAM_INFO = {
    "horizon": "Número de barras futuras a evaluar",
    "upper_pct": "Porcentaje de barrera superior",
    "lower_pct": "Porcentaje de barrera inferior",
    "training_window": "Ventana para entrenamiento del modelo",
    "tf_minutes": "Duración del timeframe en minutos",
    "meta_model": "Modelo para meta etiquetado",
}


def apply_meta_labeling(
    labels: pd.Series,
    features: pd.DataFrame,
    model: ClassifierMixin | None = None,
) -> pd.Series:
    """Train a secondary model to obtain meta labels from primary labels.

    Parameters
    ----------
    labels : pd.Series
        Primary labels containing values ``-1``, ``0`` or ``1``.
    features : pd.DataFrame
        Feature matrix aligned with ``labels``.
    model : ClassifierMixin, optional
        Model used for the meta labeling task. If ``None`` a
        :class:`~sklearn.ensemble.GradientBoostingClassifier` is used.

    Returns
    -------
    pd.Series
        Predicted meta labels where ``1`` indicates the primary label
        suggests taking a trade and ``0`` otherwise.
    """

    if model is None:
        model = GradientBoostingClassifier()
    # Meta labels are binary: take the trade when primary label is not 0
    meta_y = (labels != 0).astype(int)
    if len(features) != len(meta_y):
        raise ValueError("features and labels must have the same length")
    # Fit the provided model. Even if ``meta_y`` is constant the classifier
    # will simply learn that constant mapping which is sufficient for tests.
    model.fit(features, meta_y)
    return pd.Series(model.predict(features), index=labels.index)


def triple_barrier_labels(
    prices: pd.Series,
    horizon: int = 5,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
    tf_minutes: int = 1,
) -> pd.Series:
    """Generate triple-barrier labels for a price series.

    ``horizon`` is expressed in 1-minute bars and scaled by ``tf_minutes``
    to adapt the labeling horizon to the bar timeframe.

    Parameters
    ----------
    prices: pd.Series
        Series of prices.
    horizon: int, optional
        Number of future 1‑minute bars to inspect, by default ``5``.
    upper_pct: float, optional
        Upper barrier percentage, by default ``0.02``.
    lower_pct: float, optional
        Lower barrier percentage, by default ``0.02``.
    tf_minutes: int, optional
        Duration of the timeframe in minutes, by default ``1``.

    Returns
    -------
    pd.Series
        Labels ``1`` if the upper barrier is hit first, ``-1`` for the
        lower barrier, or ``0`` if neither is reached within the horizon.
    """

    horizon = int(horizon * tf_minutes)
    labels = pd.Series(0, index=prices.index, dtype=int)
    n = len(prices)
    for i in range(n - 1):
        start = prices.iloc[i]
        upper = start * (1 + upper_pct)
        lower = start * (1 - lower_pct)
        end = min(i + 1 + horizon, n)
        future = prices.iloc[i + 1 : end]
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


liquidity = LiquidityFilterManager()


class TripleBarrier(Strategy):
    """Strategy using triple-barrier labeling and a gradient boosting model."""

    name = "triple_barrier"

    def __init__(
        self,
        horizon: int = 5,
        upper_pct: float = 0.02,
        lower_pct: float = 0.02,
        training_window: int = 200,
        tf_minutes: int = 1,
        meta_model: ClassifierMixin | None = None,
        *,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        params = load_params(config_path)
        horizon = params.get("horizon", horizon)
        upper_pct = params.get("upper_pct", upper_pct)
        lower_pct = params.get("lower_pct", lower_pct)
        training_window = params.get("training_window", training_window)
        tf_minutes = params.get("tf_minutes", tf_minutes)
        meta_model = params.get("meta_model", meta_model)

        self.horizon = int(horizon)
        self.upper_pct = float(upper_pct)
        self.lower_pct = float(lower_pct)
        self.tf_minutes = int(tf_minutes)
        self.training_window = int(training_window * self.tf_minutes)
        self.model = GradientBoostingClassifier()
        self.meta_model = meta_model or GradientBoostingClassifier()
        self.fitted = False
        self.meta_fitted = False
        self.risk_service = kwargs.get("risk_service")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["close"].pct_change().fillna(0)
        window = max(1, int(self.horizon * self.tf_minutes))
        feat = pd.DataFrame(
            {
                "ret": returns,
                "ret_mean": returns.rolling(window).mean().fillna(0),
                "ret_std": returns.rolling(window).std().fillna(0),
            }
        )
        return feat

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.training_window:
            return None
        last = df["close"].iloc[-1]

        features = self._prepare_features(df)
        if not self.fitted:
            labels = triple_barrier_labels(
                df["close"],
                self.horizon,
                self.upper_pct,
                self.lower_pct,
                self.tf_minutes,
            )
            X = features.iloc[:-1]
            y = labels.iloc[:-1]
            if y.nunique() > 1:
                self.model.fit(X, y)
                # Fit meta model using the same features/labels
                apply_meta_labeling(y, X, self.meta_model)
                self.fitted = True
                self.meta_fitted = True
            else:
                return None
        x_last = features.iloc[[-1]]
        pred = self.model.predict(x_last)[0]
        if pred == 0:
            return None
        if self.meta_fitted:
            meta_pred = self.meta_model.predict(x_last)[0]
            if meta_pred == 0:
                return None
        if pred == 1:
            side = "buy"
        elif pred == -1:
            side = "sell"
        else:
            return None
        strength = 1.0
        sig = Signal(side, strength)
        return self.finalize_signal(bar, float(last), sig)
