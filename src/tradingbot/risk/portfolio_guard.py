# src/tradingbot/risk/portfolio_guard.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

import numpy as np

@dataclass
class GuardConfig:
    total_cap_pct: float = 1.0
    per_symbol_cap_pct: float = 0.5
    venue: str = "binance_spot_testnet"
    # --- Soft caps ---
    soft_cap_pct: float = 0.10            # hasta +10% sobre el cap
    soft_cap_grace_sec: int = 30          # por hasta 30s

@dataclass
class GuardState:
    positions: Dict[str, float] = field(default_factory=dict)   # qty base
    prices: Dict[str, float] = field(default_factory=dict)      # último mark
    # retornos simples (dprice/price_prev) para correlaciones intraportafolio
    returns: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=50))
    )
    # ventanas soft activas
    sym_soft_started: Dict[str, Optional[datetime]] = field(default_factory=dict)
    total_soft_started: Optional[datetime] = None
    # posiciones y PnL por venue
    venue_positions: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    venue_pnl: Dict[str, float] = field(default_factory=dict)
    per_symbol_cap_usdt: float = 0.0
    total_cap_usdt: float = 0.0

class PortfolioGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.st = GuardState()
        self.equity: float = 0.0

    def refresh_usd_caps(self, equity: float) -> None:
        self.equity = float(equity)
        self.st.per_symbol_cap_usdt = self.equity * self.cfg.per_symbol_cap_pct
        self.st.total_cap_usdt = self.equity * self.cfg.total_cap_pct

    # ---- utilidades base ----
    def mark_price(self, symbol: str, price: float):
        price = float(price)
        prev = self.st.prices.get(symbol)
        self.st.prices[symbol] = price
        if prev is not None and prev > 0:
            ret = (price / prev) - 1.0
            self.st.returns[symbol].append(ret)

    def update_position_on_order(self, symbol: str, side: str, qty: float, venue: str | None = None):
        cur = self.st.positions.get(symbol, 0.0)
        delta = qty if side.lower()=="buy" else -qty
        self.st.positions[symbol] = cur + delta
        if venue is not None:
            book = self.st.venue_positions.setdefault(venue, {})
            book[symbol] = book.get(symbol, 0.0) + delta

    def set_position(self, venue: str, symbol: str, qty: float) -> None:
        """Sincroniza posición absoluta para un venue."""
        self.st.positions[symbol] = qty
        book = self.st.venue_positions.setdefault(venue, {})
        book[symbol] = qty

    def exposure_symbol(self, symbol: str) -> float:
        px = self.st.prices.get(symbol, 0.0)
        pos = self.st.positions.get(symbol, 0.0)
        return abs(pos) * px

    def exposure_total(self) -> float:
        return sum(abs(self.st.positions.get(s, 0.0)) * self.st.prices.get(s, 0.0) for s in self.st.positions)

    def correlations(self) -> Dict[Tuple[str, str], float]:
        """Calcula correlaciones de retornos entre pares de símbolos."""
        syms = list(self.st.returns.keys())
        result: Dict[Tuple[str, str], float] = {}
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a = np.array(self.st.returns[syms[i]], dtype=float)
                b = np.array(self.st.returns[syms[j]], dtype=float)
                n = min(len(a), len(b))
                if n >= 2:
                    corr = float(np.corrcoef(a[-n:], b[-n:])[0, 1])
                    result[(syms[i], syms[j])] = corr
        return result

    def volatility(self, symbol: str) -> float:
        """Desviación estándar anualizada de los retornos de ``symbol``.

        Si no hay suficientes datos de retornos, retorna ``0.0``.
        """
        rets = np.array(self.st.returns.get(symbol, []), dtype=float)
        if len(rets) < 2:
            return 0.0
        return float(np.std(rets) * np.sqrt(365))

    # ---- hard caps (como antes) ----
    def would_exceed_caps(self, symbol: str, side: str, add_qty: float, price: float) -> Tuple[bool, str, dict]:
        cur_pos = self.st.positions.get(symbol, 0.0)
        new_pos = cur_pos + (add_qty if side.lower()=="buy" else -add_qty)
        sym_exp = abs(new_pos) * price

        total = 0.0
        seen = set()
        for s, pos in self.st.positions.items():
            seen.add(s)
            px = self.st.prices.get(s, 0.0)
            if s == symbol:
                total += abs(new_pos) * price
            else:
                total += abs(pos) * px
        if symbol not in seen:
            total += abs(new_pos) * price

        if sym_exp > self.st.per_symbol_cap_usdt:
            return True, f"per_symbol_cap_usdt excedido ({sym_exp:.2f} > {self.st.per_symbol_cap_usdt:.2f})", {"sym_exp": sym_exp, "total": total}
        if total > self.st.total_cap_usdt:
            return True, f"total_cap_usdt excedido ({total:.2f} > {self.st.total_cap_usdt:.2f})", {"sym_exp": sym_exp, "total": total}
        return False, "", {"sym_exp": sym_exp, "total": total}

    # ---- soft caps ----
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _soft_bounds(self) -> Tuple[float, float]:
        return (
            self.st.per_symbol_cap_usdt * (1.0 + self.cfg.soft_cap_pct),
            self.st.total_cap_usdt * (1.0 + self.cfg.soft_cap_pct),
        )

    def soft_cap_decision(self, symbol: str, side: str, add_qty: float, price: float) -> Tuple[str, str, dict]:
        """
        Devuelve:
          action: "allow" | "soft_allow" | "block"
          reason: texto
          meta:   métricas (restante, exp, etc)
        Regla:
          - Dentro de hard caps => allow
          - Entre hard y soft (<= +pct) => permite por hasta soft_cap_grace_sec (abre ventana si no existía)
          - Sobre soft cap => block inmediato
          - Si expiró la ventana => block
        """
        now = self._now()
        cur_pos = self.st.positions.get(symbol, 0.0)
        new_pos = cur_pos + (add_qty if side.lower()=="buy" else -add_qty)
        sym_exp = abs(new_pos) * price

        total = 0.0
        seen = set()
        for s, pos in self.st.positions.items():
            seen.add(s)
            px = self.st.prices.get(s, 0.0)
            if s == symbol:
                total += abs(new_pos) * price
            else:
                total += abs(pos) * px
        if symbol not in seen:
            total += abs(new_pos) * price

        per_soft, total_soft = self._soft_bounds()
        # casos
        # 1) dentro de hard caps
        if sym_exp <= self.st.per_symbol_cap_usdt and total <= self.st.total_cap_usdt:
            # si había ventanas, se cierran
            self.st.sym_soft_started.pop(symbol, None)
            if self.st.total_soft_started and total <= self.st.total_cap_usdt:
                self.st.total_soft_started = None
            return "allow", "dentro de hard caps", {"sym_exp": sym_exp, "total": total}

        # 2) excede soft (símbolo o total) -> bloqueo
        if sym_exp > per_soft or total > total_soft:
            return "block", "sobre soft cap", {"sym_exp": sym_exp, "total": total, "per_soft": per_soft, "total_soft": total_soft}

        # 3) entre hard y soft -> tolerado por ventana
        # ventana por símbolo
        sym_start = self.st.sym_soft_started.get(symbol)
        if sym_start is None:
            self.st.sym_soft_started[symbol] = now
            sym_start = now
        # ventana global
        if self.st.total_soft_started is None and total > self.st.total_cap_usdt:
            self.st.total_soft_started = now

        # tiempo restante por las ventanas activas
        rem_sym = self.cfg.soft_cap_grace_sec - (now - sym_start).total_seconds()
        rem_tot = self.cfg.soft_cap_grace_sec - (now - (self.st.total_soft_started or now)).total_seconds()
        remaining = max(0.0, min(rem_sym, rem_tot))

        if remaining > 0:
            return "soft_allow", f"soft cap activo ({remaining:.1f}s restantes)", {
                "sym_exp": sym_exp, "total": total, "remaining_sec": remaining,
                "per_soft": per_soft, "total_soft": total_soft
            }
        # expirado
        return "block", "soft cap expirado", {
            "sym_exp": sym_exp, "total": total, "remaining_sec": 0.0,
            "per_soft": per_soft, "total_soft": total_soft
        }

    # ---- auto-close qty (como antes) ----
    def compute_auto_close_qty(self, symbol: str, price: float) -> Tuple[float, str, dict]:
        pos = self.st.positions.get(symbol, 0.0)
        if abs(pos) <= 0 or price <= 0:
            return 0.0, "sin posición o precio inválido", {"pos": pos, "price": price}

        cur_sym_exp = abs(pos) * price
        target_sym_exp = min(cur_sym_exp, self.st.per_symbol_cap_usdt)
        if cur_sym_exp > self.st.per_symbol_cap_usdt:
            need_exp_cut = cur_sym_exp - target_sym_exp
            need_qty_cut = need_exp_cut / price
        else:
            need_qty_cut = 0.0

        total = self.exposure_total()
        if total > self.st.total_cap_usdt:
            extra = total - self.st.total_cap_usdt
            need_qty_cut += extra / price

        need_qty_cut = max(0.0, min(abs(pos), need_qty_cut))
        if need_qty_cut <= 0:
            return 0.0, "dentro de límites", {"pos": pos, "price": price, "total": total}
        msg = f"auto-close {symbol}: recortar {need_qty_cut:.6f} (pos={pos:.6f}, px={price:.2f})"
        return need_qty_cut, msg, {"pos": pos, "price": price, "total": total}
