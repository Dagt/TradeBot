from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class GuardConfig:
    total_cap_usdt: float = 1000.0
    per_symbol_cap_usdt: float = 500.0
    venue: str = "binance_spot_testnet"

@dataclass
class GuardState:
    positions: Dict[str, float] = field(default_factory=dict)   # qty base por símbolo
    prices: Dict[str, float] = field(default_factory=dict)      # último precio conocido por símbolo

class PortfolioGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.st = GuardState()

    def mark_price(self, symbol: str, price: float):
        self.st.prices[symbol] = float(price)

    def update_position_on_order(self, symbol: str, side: str, qty: float):
        cur = self.st.positions.get(symbol, 0.0)
        self.st.positions[symbol] = cur + (qty if side.lower()=="buy" else -qty)

    def exposure_symbol(self, symbol: str) -> float:
        px = self.st.prices.get(symbol, 0.0)
        pos = self.st.positions.get(symbol, 0.0)
        return abs(pos) * px

    def exposure_total(self) -> float:
        return sum(abs(self.st.positions.get(s, 0.0)) * self.st.prices.get(s, 0.0) for s in self.st.positions)

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

        if sym_exp > self.cfg.per_symbol_cap_usdt:
            return True, f"per_symbol_cap_usdt excedido ({sym_exp:.2f} > {self.cfg.per_symbol_cap_usdt:.2f})", {"sym_exp": sym_exp, "total": total}
        if total > self.cfg.total_cap_usdt:
            return True, f"total_cap_usdt excedido ({total:.2f} > {self.cfg.total_cap_usdt:.2f})", {"sym_exp": sym_exp, "total": total}
        return False, "", {"sym_exp": sym_exp, "total": total}

    def compute_auto_close_qty(self, symbol: str, price: float) -> Tuple[float, str, dict]:
        """
        Devuelve (qty_a_cerrar, mensaje, métricas) para volver dentro de caps.
        Regla: primero trae el símbolo debajo de per_symbol_cap, luego el total.
        Si no hay posición o px=0, retorna (0, ...).
        """
        pos = self.st.positions.get(symbol, 0.0)
        if abs(pos) <= 0 or price <= 0:
            return 0.0, "sin posición o precio inválido", {"pos": pos, "price": price}

        # 1) per-symbol
        cur_sym_exp = abs(pos) * price
        target_sym_exp = min(cur_sym_exp, self.cfg.per_symbol_cap_usdt)
        if cur_sym_exp > self.cfg.per_symbol_cap_usdt:
            need_exp_cut = cur_sym_exp - target_sym_exp
            need_qty_cut = need_exp_cut / price
        else:
            need_qty_cut = 0.0

        # 2) total
        total = self.exposure_total()
        if total > self.cfg.total_cap_usdt:
            extra = total - self.cfg.total_cap_usdt
            # recorta adicionalmente en este símbolo
            need_qty_cut += extra / price

        need_qty_cut = max(0.0, min(abs(pos), need_qty_cut))
        if need_qty_cut <= 0:
            return 0.0, "dentro de límites", {"pos": pos, "price": price, "total": total}
        msg = f"auto-close {symbol}: recortar {need_qty_cut:.6f} (pos={pos:.6f}, px={price:.2f})"
        return need_qty_cut, msg, {"pos": pos, "price": price, "total": total}
