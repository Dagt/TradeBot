"""Funciones auxiliares para dimensionar posiciones."""

from __future__ import annotations


def vol_target(atr: float, risk_budget: float, notional_cap: float) -> float:
    """Calcula tamaño de posición basado en volatilidad (ATR).

    Parameters
    ----------
    atr: float
        Medida de volatilidad actual (p.ej. ATR).
    risk_budget: float
        Riesgo monetario asignado para la operación.
    notional_cap: float
        Exposición nominal máxima permitida.

    Returns
    -------
    float
        Tamaño de posición absoluto que respeta ``notional_cap``.
    """
    if atr <= 0:
        return 0.0
    atr = float(atr)
    risk_budget = abs(risk_budget)
    notional_cap = abs(notional_cap)
    size = risk_budget / atr
    return min(notional_cap, size)

