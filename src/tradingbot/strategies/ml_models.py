import numpy as np
from typing import Any
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from .base import Strategy, Signal, record_signal_metrics


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
        threshold: float = 0.5,
        *,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
    ) -> None:
        self.model: BaseEstimator | None = model
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.tp_bps = float(tp_bps)
        self.sl_bps = float(sl_bps)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0
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
        buy = proba >= self.threshold
        sell = proba <= 1 - self.threshold
        price = bar.get("price") or bar.get("close")
        price = float(price) if price is not None else None

        if self.pos_side == 0:
            if buy:
                self.pos_side = 1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("buy", proba)
            if sell:
                self.pos_side = -1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("sell", 1 - proba)
            return None

        self.hold_bars += 1
        exit_signal = (sell and self.pos_side > 0) or (buy and self.pos_side < 0)
        exit_tp = exit_sl = False
        if price is not None and self.entry_price is not None:
            pnl_bps = (
                (price - self.entry_price) / self.entry_price * 10000 * self.pos_side
            )
            exit_tp = pnl_bps >= self.tp_bps
            exit_sl = pnl_bps <= -self.sl_bps
        exit_time = self.hold_bars >= self.max_hold_bars
        if exit_signal or exit_tp or exit_sl or exit_time:
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None


__all__ = ["MLStrategy"]
