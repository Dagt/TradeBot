import numpy as np
from typing import Any
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from .base import Strategy, Signal, record_signal_metrics


PARAM_INFO = {
    "model": "Instancia de modelo sklearn preentrenado",
    "model_path": "Ruta para cargar el modelo",
    "margin": "Margen de probabilidad sobre 0.5",
}

class MLStrategy(Strategy):
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
        margin: float = 0.1,
        risk_service=None,
    ) -> None:
        self.model: BaseEstimator | None = model
        self.scaler = StandardScaler()
        self.margin = float(margin)
        self.risk_service = risk_service
        self.trade: dict | None = None
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
        X = np.asarray(feats, dtype=float).reshape(1, -1)
        try:
            X_scaled = self.scaler.transform(X)
            proba = float(self.model.predict_proba(X_scaled)[0, 1])
        except NotFittedError:
            return None
        proba = max(0.0, min(1.0, proba))
        price = float(bar.get("close") or bar.get("price") or 0.0)
        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, price)
            trade = {**self.trade, "current_price": price}
            decision = self.risk_service.manage_position(trade)
            self.trade.update(trade)
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            if decision in {"scale_in", "scale_out"}:
                return Signal(self.trade["side"], self.trade.get("strength", 1.0))
            return None

        if proba > 0.5 + self.margin:
            side = "buy"
            strength = proba
        elif proba < 0.5 - self.margin:
            side = "sell"
            strength = 1 - proba
        else:
            return None

        if self.risk_service:
            qty = self.risk_service.calc_position_size(strength, price)
            trade = {"side": side, "entry_price": price, "qty": qty, "strength": strength}
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(price, side, atr)
            if atr is not None:
                trade["atr"] = atr
            self.risk_service.update_trailing(trade, price)
            self.trade = trade
        return Signal(side, strength)


__all__ = ["MLStrategy"]
