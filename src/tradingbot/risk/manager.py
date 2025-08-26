"""Risk manager with event-based integration.

This module started as a very small utility to gate orders based on
simple stop loss and drawdown thresholds.  The exercises for this kata
require extending it with a few extra features:

* sizing based on a target volatility
* daily loss / drawdown stops ("kill-switch")
* correlation limits derived from a covariance matrix
* integration with strategies and execution through an :class:`EventBus`

The implementation purposely keeps the original public API so existing
tests and examples continue to function, while adding optional event
driven behaviour when a bus is supplied.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from tradingbot.utils.metrics import RISK_EVENTS, KILL_SWITCH_ACTIVE
from ..bus import EventBus
from .limits import RiskLimits, LimitTracker
from .position_sizing import vol_target, delta_from_strength
from sqlalchemy import text


@dataclass
class Position:
    """Simple container for current position size."""

    qty: float = 0.0


class StopLossExceeded(Exception):
    """Raised when unrealized losses breach the configured ``risk_pct``."""


class RiskManager:
    """Gestor de riesgo mínimo.

    Añade límites de pérdida y drawdown y un *kill switch* mediante ``enabled``.
    ``check_limits`` debe invocarse con el precio de mercado actual antes de
    enviar cualquier orden.  Si los límites se violan, ``enabled`` se vuelve
    ``False`` y no se deben continuar enviando órdenes.
    """

    def __init__(
        self,
        risk_pct: float = 0.0,
        vol_target: float = 0.0,
        *,
        daily_loss_limit: float = 0.0,
        daily_drawdown_pct: float = 0.0,
        limits: RiskLimits | None = None,
        bus: EventBus | None = None,
    ):
        """Create a :class:`RiskManager`.

        ``daily_loss_limit`` and ``daily_drawdown_pct`` enable intraday halt
        rules.  When ``bus`` is provided, the manager subscribes to a few topics
        in order to keep its state in sync with fills and price marks and to
        emit ``risk:halted`` events when a limit is breached.
        """

        self.risk_pct = abs(risk_pct)
        self.vol_target = abs(vol_target)
        self._base_vol_target = self.vol_target

        self.daily_loss_limit = abs(daily_loss_limit)
        self.daily_drawdown_pct = abs(daily_drawdown_pct)
        self.daily_pnl: float = 0.0
        self._daily_peak: float = 0.0

        self.enabled = True
        self.last_kill_reason: str | None = None
        self.pos = Position()
        # Multi-exchange position book: {exchange: {symbol: qty}}
        self.positions_multi: Dict[str, Dict[str, float]] = defaultdict(dict)
        # PnL por exchange
        self.pnl_multi: Dict[str, float] = defaultdict(float)
        # Flag metric for kill switch status
        KILL_SWITCH_ACTIVE.set(0)
        # Variable interna para cálculo de SL
        self._entry_price: float | None = None

        self.bus = bus
        if self.bus is not None:
            # Actualiza posición y precios desde la ejecución
            self.bus.subscribe("fill", self._on_fill_event)
            self.bus.subscribe("price", self._on_price_event)
            self.bus.subscribe("pnl", self._on_pnl_event)
            self.bus.subscribe("risk:blocked", self._on_block_event)

        self.limits = LimitTracker(limits, bus) if limits is not None else None

    def _reset_price_trackers(self) -> None:
        self._entry_price = None

    def set_position(self, qty: float) -> None:
        """Sincroniza la posición mantenida por el RiskManager con el qty real."""
        self.pos.qty = float(qty)
        if abs(self.pos.qty) < 1e-12:
            self._reset_price_trackers()

    def add_fill(self, side: str, qty: float) -> None:
        """Actualiza posición interna tras un fill."""
        signed = float(qty) if side == "buy" else -float(qty)
        prev = self.pos.qty
        self.pos.qty += signed
        # Si se cerró o se invirtió la posición, reinicia trackers de precio
        if prev == 0 or prev * self.pos.qty <= 0:
            self._reset_price_trackers()

    # ------------------------------------------------------------------
    # Order limit helpers
    def register_order(self, notional: float) -> bool:
        """Registrar una nueva orden verificando límites opcionales."""
        if self.limits is None:
            return True
        return self.limits.register_order(notional)

    def complete_order(self) -> None:
        if self.limits is not None:
            self.limits.complete_order()

    # ------------------------------------------------------------------
    # Multi-exchange position helpers
    def update_position(self, exchange: str, symbol: str, qty: float) -> None:
        """Actualizar posición por exchange y símbolo."""
        book = self.positions_multi.setdefault(exchange, {})
        book[symbol] = float(qty)

    def aggregate_positions(self) -> Dict[str, float]:
        """Agrega posiciones de todos los exchanges por símbolo."""
        agg: Dict[str, float] = {}
        for book in self.positions_multi.values():
            for sym, qty in book.items():
                agg[sym] = agg.get(sym, 0.0) + float(qty)
        return agg

    def close_all_positions(self) -> None:
        """Resetea todas las posiciones (locales y agregadas)."""
        self.pos.qty = 0.0
        for exch in list(self.positions_multi.keys()):
            for sym in list(self.positions_multi[exch].keys()):
                self.positions_multi[exch][sym] = 0.0

    def covariance_matrix(self, returns: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
        """Calcula matriz de covarianza a partir de retornos por símbolo."""
        if not returns:
            return {}
        syms = list(returns.keys())
        arrays = [np.array(returns[s], dtype=float) for s in syms]
        n = min((len(a) for a in arrays), default=0)
        if n == 0:
            return {}
        if len(syms) == 1:
            arr = arrays[0][-n:]
            ddof = 1 if arr.size > 1 else 0
            cov = np.var(arr, ddof=ddof)
            return {(syms[0], syms[0]): float(cov)}
        data = np.stack([a[-n:] for a in arrays])
        cov = np.cov(data)
        mat: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(syms):
            for j, b in enumerate(syms):
                mat[(a, b)] = float(cov[i, j])
        return mat

    def adjust_size_for_correlation(
        self,
        symbol: str,
        base_size: float,
        correlations: Dict[Tuple[str, str], float],
        threshold: float,
    ) -> float:
        """Reduce tamaño de orden si hay correlaciones altas con otras posiciones."""
        for (a, b), corr in correlations.items():
            if symbol in (a, b) and abs(corr) >= threshold:
                RISK_EVENTS.labels(event_type="correlated_size_cut").inc()
                return base_size * 0.5
        return base_size

    # ------------------------------------------------------------------
    # Event bus callbacks
    async def _on_fill_event(self, evt: dict) -> None:
        side = evt.get("side", "")
        qty = evt.get("qty", 0.0)
        venue = evt.get("venue")
        symbol = evt.get("symbol")
        self.add_fill(side, qty)
        if venue and symbol:
            book = self.positions_multi.setdefault(venue, {})
            signed = float(qty) if side == "buy" else -float(qty)
            book[symbol] = book.get(symbol, 0.0) + signed

    async def _on_price_event(self, evt: dict) -> None:
        price = evt.get("price")
        if price is None:
            return
        if not self.check_limits(price) and self.bus is not None:
            await self.bus.publish("risk:halted", {"reason": self.last_kill_reason})

    async def _on_pnl_event(self, evt: dict) -> None:
        delta = float(evt.get("delta", 0.0))
        venue = evt.get("venue")
        self.update_pnl(delta, venue=venue)
        if not self._check_daily_limits() and self.bus is not None:
            await self.bus.publish("risk:halted", {"reason": self.last_kill_reason})

    async def _on_block_event(self, evt: dict) -> None:
        reason = evt.get("reason", "limit")
        self.enabled = False
        self.last_kill_reason = reason
        self.close_all_positions()

    def size(
        self,
        signal_side: str,
        price: float,
        equity: float,
        strength: float = 1.0,
        *,
        symbol: str | None = None,
        symbol_vol: float = 0.0,
        correlations: Dict[Tuple[str, str], float] | None = None,
        threshold: float = 0.0,
    ) -> float:
        """Compute position delta for a strength-based signal.

        El tamaño se obtiene con ``notional = equity * strength``. ``price`` es
        el precio del activo y ``equity`` el capital actual. Valores de
        ``strength`` mayores a ``1`` piramidan la posición y menores la
        reducen. El stop‑loss local se calcula aparte como ``notional * risk_pct``.
        """

        signed_strength = strength if signal_side == "buy" else -strength
        delta = delta_from_strength(signed_strength, equity, price, self.pos.qty)

        if symbol_vol > 0:
            delta += self.size_with_volatility(symbol_vol, price, equity)

        if correlations:
            if symbol is None:
                raise ValueError("symbol requerido para aplicar correlaciones")
            delta = self.adjust_size_for_correlation(symbol, delta, correlations, threshold)

        return delta

    def check_order(
        self,
        symbol: str,
        side: str,
        equity: float,
        price: float,
        *,
        strength: float = 1.0,
        symbol_vol: float = 0.0,
        correlations: Dict[Tuple[str, str], float] | None = None,
        threshold: float = 0.0,
    ) -> tuple[bool, str, float]:
        """Calcular tamaño y validar límites para una orden."""

        delta = self.size(
            side,
            price,
            equity,
            strength,
            symbol=symbol,
            symbol_vol=symbol_vol,
            correlations=correlations,
            threshold=threshold,
        )

        if abs(delta) <= 0:
            return False, "zero_size", 0.0

        try:
            if not self.check_limits(price):
                return False, "kill_switch", 0.0
        except StopLossExceeded:
            return False, "stop_loss", 0.0
        return True, "", delta

    def size_with_volatility(self, symbol_vol: float, price: float, equity: float) -> float:
        """Dimensiona la posición basada en un objetivo de volatilidad.

        Devuelve el delta necesario para alcanzar el objetivo usando
        :func:`delta_from_strength` para mantener piramidado consistente.
        """
        if self.vol_target <= 0 or symbol_vol <= 0:
            return 0.0

        target_abs = vol_target(symbol_vol, self.vol_target, equity)
        sign = 1 if self.pos.qty >= 0 else -1
        target = sign * target_abs
        if equity <= 0 or self.vol_target <= 0 or price <= 0:
            strength = 0.0
        else:
            strength = target * price / equity
        strength = max(-1.0, min(1.0, strength))
        RISK_EVENTS.labels(event_type="volatility_sizing").inc()
        return delta_from_strength(strength, equity, price, self.pos.qty)

    def update_correlation(
        self, symbol_pairs: dict[tuple[str, str], float], threshold: float
    ) -> list[tuple[str, str]]:
        """Limita exposición conjunta reduciendo ``vol_target`` si se supera el umbral.

        Los símbolos con correlaciones que exceden ``threshold`` se agrupan
        mediante :mod:`correlation_guard` y ``vol_target`` se reduce en proporción
        al tamaño del grupo más grande.  Se devuelve la lista de pares que
        superaron el umbral.
        """

        from .correlation_guard import group_correlated, global_cap

        exceeded: list[tuple[str, str]] = [
            pair for pair, corr in symbol_pairs.items() if abs(corr) >= threshold
        ]

        groups = group_correlated(symbol_pairs, threshold)
        self.vol_target = global_cap(groups, self._base_vol_target)

        if exceeded:
            RISK_EVENTS.labels(event_type="correlation_limit").inc()
            if self.bus is not None:
                asyncio.create_task(
                    self.bus.publish(
                        "risk:paused", {"reason": "correlation", "pairs": exceeded}
                    )
                )
        return exceeded

    def update_covariance(
        self, cov_matrix: Dict[tuple[str, str], float], threshold: float
    ) -> list[tuple[str, str]]:
        r"""Aplica un límite de correlación a partir de una matriz de covarianzas.

        Se calcula la correlación \rho_{i,j} = cov_{i,j} / (\sigma_i \sigma_j)
        para cada par ``(i, j)``.  Si el valor absoluto excede ``threshold`` se
        recorta ``vol_target`` a la mitad y se retornan los pares involucrados.
        """

        exceeded: list[tuple[str, str]] = []
        vars: Dict[str, float] = {}
        for (a, b), cov in cov_matrix.items():
            if a == b:
                vars[a] = abs(cov)
        for (a, b), cov in cov_matrix.items():
            if a == b:
                continue
            va = vars.get(a)
            vb = vars.get(b)
            if va and vb and va > 0 and vb > 0:
                corr = cov / ((va ** 0.5) * (vb ** 0.5))
                if abs(corr) >= threshold:
                    exceeded.append((a, b))
        if exceeded:
            self.vol_target *= 0.5
            RISK_EVENTS.labels(event_type="correlation_limit").inc()
            if self.bus is not None:
                asyncio.create_task(
                    self.bus.publish(
                        "risk:paused", {"reason": "covariance", "pairs": exceeded}
                    )
                )
        return exceeded

    def check_portfolio_risk(
        self,
        positions: Dict[str, float],
        cov_matrix: Dict[tuple[str, str], float],
        max_variance: float,
    ) -> bool:
        """Evalúa el riesgo total mediante una matriz de covarianzas.

        ``positions`` contiene cantidades por símbolo (mismas unidades que la
        covarianza).  Si la varianza total del portafolio excede
        ``max_variance`` la función retorna ``False``.
        """

        if not positions:
            return True
        syms = list(positions.keys())
        n = len(syms)
        # Construye matriz completa
        mat = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(syms):
            for j, b in enumerate(syms):
                mat[i][j] = float(cov_matrix.get((a, b), cov_matrix.get((b, a), 0.0)))
        # p^T Σ p
        var = 0.0
        for i in range(n):
            for j in range(n):
                var += positions[syms[i]] * mat[i][j] * positions[syms[j]]
        if var > max_variance:
            self.last_kill_reason = "covariance_limit"
            return False
        return True

    def kill_switch(self, reason: str) -> None:
        """Deshabilita completamente el gestor de riesgo."""
        self.enabled = False
        self.last_kill_reason = reason
        # close any tracked positions
        self.close_all_positions()
        RISK_EVENTS.labels(event_type="kill_switch").inc()
        KILL_SWITCH_ACTIVE.set(1)
        if self.bus is not None:
            # dispatch asynchronously without awaiting
            asyncio.create_task(self.bus.publish("risk:halted", {"reason": reason}))

    # ------------------------------------------------------------------
    # Daily PnL / Drawdown tracking
    def update_pnl(self, delta: float, *, venue: str | None = None) -> None:
        """Actualizar PnL diario y por venue."""
        self.daily_pnl += float(delta)
        if venue:
            self.pnl_multi[venue] = self.pnl_multi.get(venue, 0.0) + float(delta)
        if self.daily_pnl > self._daily_peak:
            self._daily_peak = self.daily_pnl
        if self.limits is not None:
            self.limits.update_pnl(delta)

    def _check_daily_limits(self) -> bool:
        if self.daily_loss_limit > 0 and self.daily_pnl <= -self.daily_loss_limit:
            self.kill_switch("daily_loss")
            return False
        if (
            self.daily_drawdown_pct > 0
            and self._daily_peak > 0
            and (self._daily_peak - self.daily_pnl) / self._daily_peak >= self.daily_drawdown_pct
        ):
            self.kill_switch("daily_drawdown")
            return False
        return True

    def check_limits(self, price: float) -> bool:
        """Verifica stop loss basado en ``risk_pct``.

        Cuando la pérdida latente excede ``notional * risk_pct`` se
        lanza :class:`StopLossExceeded`.
        """

        if not self.enabled:
            return False
        if self.limits is not None and self.limits.blocked:
            return False

        qty = self.pos.qty
        if abs(qty) < 1e-12:
            self._reset_price_trackers()
            return True

        px = float(price)
        if self._entry_price is None:
            self._entry_price = px
            return True

        notional = abs(qty) * self._entry_price
        pnl = (px - self._entry_price) * qty
        if self.risk_pct > 0 and pnl < -notional * self.risk_pct:
            self.last_kill_reason = "stop_loss"
            raise StopLossExceeded("stop_loss")

        return self._check_daily_limits()


# ---------------------------------------------------------------------------
# Rehidratación desde base de datos
def load_positions(engine, venue: str) -> dict:
    """Cargar posiciones persistidas para ``venue``.

    Devuelve un diccionario ``{symbol: {qty, avg_price, ...}}`` obtenido de la
    capa de almacenamiento.  Si ``engine`` es ``None`` o ocurre algún error se
    retorna un mapeo vacío para evitar que la carga falle en entornos sin base
    de datos.
    """

    if engine is None:
        return {}
    try:  # pragma: no cover - si faltan dependencias simplemente ignoramos
        from ..storage.timescale import load_positions as _load
        return _load(engine, venue)
    except Exception:
        try:  # fallback para motores sin esquema Timescale (ej. SQLite)
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        'SELECT venue, symbol, qty, avg_price, realized_pnl, fees_paid'
                        ' FROM "market.positions" WHERE venue = :venue'
                    ),
                    {"venue": venue},
                ).mappings().all()
            out: dict = {}
            for r in rows:
                out[r["symbol"]] = {
                    "qty": float(r["qty"]),
                    "avg_price": float(r["avg_price"]),
                    "realized_pnl": float(r["realized_pnl"]),
                    "fees_paid": float(r["fees_paid"]),
                }
            return out
        except Exception:
            return {}

