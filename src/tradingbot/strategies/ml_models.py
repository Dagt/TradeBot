import numpy as np
from typing import Any
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from .base import StatefulStrategy, Signal, record_signal_metrics


class MLStrategy(StatefulStrategy):
    """Machine learning based strategy using scikit-learn models.

    The strategy generates buy/sell signals from feature vectors present in the
    ``bar`` dictionary under the ``features`` key. A model can either be loaded
    from disk or trained from data provided at construction time.
    """

    name = "ml"

    def __init__(
        self,
        model: BaseEstimator | None = None,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        *,
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
        max_hold_bars: int = 0,
    ) -> None:
        super().__init__(tp_pct=tp_pct, sl_pct=sl_pct, max_hold_bars=max_hold_bars)
        self.model: BaseEstimator | None = model
        self.scaler = StandardScaler()
        self.threshold = threshold
        if model_path:
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Model persistence
    def load_model(self, path: str | Path) -> None:
        """Load a trained model (and optional scaler) from ``path``."""
        obj = load(path)
        if isinstance(obj, tuple):
            self.model, self.scaler = obj
        else:
            self.model = obj

    def save_model(self, path: str | Path) -> None:
        """Persist the current model and scaler to ``path``."""
        if self.model is None:
            raise ValueError("No model to save")
        dump((self.model, self.scaler), path)

    # ------------------------------------------------------------------
    # Training
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a ``LogisticRegression`` model on the provided data."""
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = LogisticRegression()
        self.model.fit(X_scaled, y)

    # ------------------------------------------------------------------
    @record_signal_metrics
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        """Generate a signal for a given bar using the ML model.

        Parameters
        ----------
        bar : dict
            Must contain a ``features`` key with an iterable of numeric
            values representing the feature vector for the current bar.
        """
        if self.model is None:
            return None
        feats = bar.get("features")
        if feats is None:
            return None
        price = bar.get("price")
        if price is not None:
            exit_sig = self.check_exit(float(price))
            if exit_sig:
                return exit_sig
        X = np.asarray(feats, dtype=float).reshape(1, -1)
        try:
            X_scaled = self.scaler.transform(X)
            proba = float(self.model.predict_proba(X_scaled)[0, 1])
        except NotFittedError:
            return None
        if self.pos_side is None and price is not None:
            if proba >= self.threshold:
                return self.open_position("buy", float(price))
            if proba <= 1 - self.threshold:
                return self.open_position("sell", float(price))
        return None


__all__ = ["MLStrategy"]
