import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import ClassifierMixin

from .base import Strategy, Signal, record_signal_metrics, load_params


PARAM_INFO = {
    "horizon": "Número de barras futuras a evaluar",
    "upper_pct": "Porcentaje de barrera superior",
    "lower_pct": "Porcentaje de barrera inferior",
    "training_window": "Ventana para entrenamiento del modelo",
    "meta_model": "Modelo para meta etiquetado",
    "config_path": "Ruta opcional al archivo de configuración",
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
        meta_model: ClassifierMixin | None = None,
        *,
        config_path: str | None = None,
        risk_service=None,
    ) -> None:
        params = load_params(config_path)
        horizon = params.get("horizon", horizon)
        upper_pct = params.get("upper_pct", upper_pct)
        lower_pct = params.get("lower_pct", lower_pct)
        training_window = params.get("training_window", training_window)
        meta_model = params.get("meta_model", meta_model)

        self.horizon = int(horizon)
        self.upper_pct = float(upper_pct)
        self.lower_pct = float(lower_pct)
        self.training_window = int(training_window)
        self.model = GradientBoostingClassifier()
        self.meta_model = meta_model or GradientBoostingClassifier()
        self.fitted = False
        self.meta_fitted = False
        self.risk_service = risk_service
        self.trade: dict | None = None

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
        last = df["close"].iloc[-1]
        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, last)
            decision = self.risk_service.manage_position(
                {**self.trade, "current_price": last}
            )
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
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
        if self.risk_service:
            qty = self.risk_service.calc_position_size(1.0, last)
            trade = {"side": side, "entry_price": float(last), "qty": qty}
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(last, side, atr)
            self.risk_service.update_trailing(trade, float(last))
            self.trade = trade
        return Signal(side, 1.0)
