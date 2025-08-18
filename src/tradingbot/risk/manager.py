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


@dataclass
class Position:
    """Simple container for current position size."""

    qty: float = 0.0


class RiskManager:
    """Gestor de riesgo mínimo.

    Añade límites de pérdida y drawdown y un *kill switch* mediante ``enabled``.
    ``check_limits`` debe invocarse con el precio de mercado actual antes de
    enviar cualquier orden.  Si los límites se violan, ``enabled`` se vuelve
    ``False`` y no se deben continuar enviando órdenes.
    """

    def __init__(
        self,
        max_pos: float = 1.0,
        stop_loss_pct: float = 0.0,
        max_drawdown_pct: float = 0.0,
        vol_target: float = 0.0,
        *,
        daily_loss_limit: float = 0.0,
        daily_drawdown_pct: float = 0.0,
        limits: RiskLimits | None = None,
        bus: EventBus | None = None,
    ):
        """Create a :class:`RiskManager`.

        Parameters mirror the previous version.  ``daily_loss_limit`` and
        ``daily_drawdown_pct`` enable intraday halt rules.  When ``bus`` is
        provided, the manager subscribes to a few topics in order to keep its
        state in sync with fills and price marks and to emit ``risk:halted``
        events when a limit is breached.
        """

        self.max_pos = max_pos
        self.stop_loss_pct = abs(stop_loss_pct)
        self.max_drawdown_pct = abs(max_drawdown_pct)
        self.vol_target = abs(vol_target)
        self._base_max_pos = self.max_pos
        self._base_vol_target = self.vol_target
        self._de_risk_stage = 0

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
        # Variables internas para cálculo de SL/DD
        self._entry_price: float | None = None
        self._peak_price: float | None = None  # máximo a favor (long) / mínimo a favor (short)
        self._trough_price: float | None = None

        self.bus = bus
        if self.bus is not None:
            # Actualiza posición y precios desde la ejecución
            self.bus.subscribe("fill", self._on_fill_event)
            self.bus.subscribe("price", self._on_price_event)
            self.bus.subscribe("pnl", self._on_pnl_event)

        self.limits = LimitTracker(limits, bus) if limits is not None else None

    def _reset_price_trackers(self) -> None:
        self._entry_price = None
        self._peak_price = None
        self._trough_price = None

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

    def size(
        self,
        signal_side: str,
        strength: float = 1.0,
        *,
        symbol: str | None = None,
        symbol_vol: float = 0.0,
        correlations: Dict[Tuple[str, str], float] | None = None,
        threshold: float = 0.0,
    ) -> float:
        """Compute desired position delta with optional volatility and correlation adjustments.

        Parameters
        ----------
        signal_side:
            "buy" para posición larga, "sell" para corta.
        strength:
            Factor para escalar la posición objetivo.
        symbol:
            Símbolo del activo (necesario si se aplican correlaciones).
        symbol_vol:
            Volatilidad actual del símbolo para ajustar al objetivo ``vol_target``.
        correlations:
            Mapa de correlaciones {(sym_a, sym_b): rho} para limitar tamaño.
        threshold:
            Umbral de correlación sobre el cual se reduce la posición.
        """

        target = self.max_pos * (
            1 if signal_side == "buy" else -1 if signal_side == "sell" else 0
        )
        delta = (target - self.pos.qty) * strength

        if symbol_vol > 0:
            delta += self.size_with_volatility(symbol_vol)

        if correlations:
            if symbol is None:
                raise ValueError("symbol requerido para aplicar correlaciones")
            delta = self.adjust_size_for_correlation(symbol, delta, correlations, threshold)

        return delta

    def size_with_volatility(self, symbol_vol: float) -> float:
        """Dimensiona la posición basada en un objetivo de volatilidad.

        Si ``vol_target`` o ``symbol_vol`` no son positivos, no se ajusta tamaño.
        Devuelve el delta necesario para alcanzar el objetivo.
        """
        if self.vol_target <= 0 or symbol_vol <= 0:
            return 0.0
        scale = self.vol_target / symbol_vol
        target_abs = min(self.max_pos, self.max_pos * scale)
        sign = 1 if self.pos.qty >= 0 else -1
        target = sign * target_abs
        RISK_EVENTS.labels(event_type="volatility_sizing").inc()
        return target - self.pos.qty

    def update_correlation(
        self, symbol_pairs: dict[tuple[str, str], float], threshold: float
    ) -> list[tuple[str, str]]:
        """Limita exposición conjunta reduciendo ``max_pos`` si se supera el umbral.

        Los símbolos con correlaciones que exceden ``threshold`` se agrupan
        mediante :mod:`correlation_guard` y ``max_pos`` se reduce en proporción
        al tamaño del grupo más grande.  Se devuelve la lista de pares que
        superaron el umbral.
        """

        from .correlation_guard import group_correlated, global_cap

        exceeded: list[tuple[str, str]] = [
            pair for pair, corr in symbol_pairs.items() if abs(corr) >= threshold
        ]

        groups = group_correlated(symbol_pairs, threshold)
        self.max_pos = global_cap(groups, self._base_max_pos)

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
        recorta ``max_pos`` a la mitad y se retornan los pares involucrados.
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
            self.max_pos *= 0.5
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
        covarianza).  Si la varianza total del portafolio excede ``max_variance``
        se dispara el *kill switch* y la función retorna ``False``.
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
            self.kill_switch("covariance_limit")
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

    def reset(self) -> None:
        """Reinicia el gestor tras una activación del *kill switch*."""
        self.enabled = True
        self.last_kill_reason = None
        self.close_all_positions()
        self.positions_multi.clear()
        self.pnl_multi.clear()
        self.daily_pnl = 0.0
        self._daily_peak = 0.0
        self.max_pos = self._base_max_pos
        self.vol_target = self._base_vol_target
        self._de_risk_stage = 0
        self._reset_price_trackers()
        KILL_SWITCH_ACTIVE.set(0)
        if self.limits is not None:
            self.limits.reset()

    # ------------------------------------------------------------------
    # Daily PnL / Drawdown tracking
    def de_risk(self) -> None:
        """Reduce ``max_pos`` y ``vol_target`` ante drawdowns significativos."""
        if self._daily_peak <= 0:
            return
        drawdown = (self._daily_peak - self.daily_pnl) / self._daily_peak
        if drawdown >= 0.5 and self._de_risk_stage < 2:
            self.max_pos = self._base_max_pos * 0.25
            if self._base_vol_target > 0:
                self.vol_target = self._base_vol_target * 0.25
            self._de_risk_stage = 2
        elif drawdown >= 0.25 and self._de_risk_stage < 1:
            self.max_pos = self._base_max_pos * 0.5
            if self._base_vol_target > 0:
                self.vol_target = self._base_vol_target * 0.5
            self._de_risk_stage = 1

    def update_pnl(self, delta: float, *, venue: str | None = None) -> None:
        """Actualizar PnL diario y por venue."""
        self.daily_pnl += float(delta)
        if venue:
            self.pnl_multi[venue] = self.pnl_multi.get(venue, 0.0) + float(delta)
        if self.daily_pnl > self._daily_peak:
            self._daily_peak = self.daily_pnl
        self.de_risk()
        if self.limits is not None:
            self.limits.update_pnl(delta)

    def _check_daily_limits(self) -> bool:
        self.de_risk()
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
        """Verifica stop loss y drawdown.

        Actualiza precios de referencia y retorna ``True`` si aún se pueden
        operar órdenes.  Si un límite se viola, ``enabled`` pasa a ``False`` y
        la función devuelve ``False``.
        """

        if not self.enabled:
            return False
        if self.limits is not None and self.limits.blocked:
            return False

        qty = self.pos.qty
        if abs(qty) < 1e-12:
            # Sin posición -> resetea referencias
            self._reset_price_trackers()
            return True

        px = float(price)
        if self._entry_price is None:
            # Primer precio tras abrir posición
            self._entry_price = px
            self._peak_price = px
            self._trough_price = px
            return True

        if qty > 0:  # posición larga
            if px > (self._peak_price or px):
                self._peak_price = px
            if px < (self._trough_price or px):
                self._trough_price = px
            loss_pct = (self._entry_price - px) / self._entry_price if self.stop_loss_pct > 0 else 0.0
            drawdown = (
                (self._peak_price - px) / self._peak_price if self.max_drawdown_pct > 0 and self._peak_price else 0.0
            )
        else:  # posición corta
            if px < (self._trough_price or px):
                self._trough_price = px
            if px > (self._peak_price or px):
                self._peak_price = px
            loss_pct = (px - self._entry_price) / self._entry_price if self.stop_loss_pct > 0 else 0.0
            drawdown = (
                (px - self._trough_price) / self._trough_price if self.max_drawdown_pct > 0 and self._trough_price else 0.0
            )

        if self.stop_loss_pct > 0 and loss_pct >= self.stop_loss_pct:
            self.kill_switch("stop_loss")
            return False
        if self.max_drawdown_pct > 0 and drawdown >= self.max_drawdown_pct:
            self.kill_switch("drawdown")
            return False

        return self._check_daily_limits()
