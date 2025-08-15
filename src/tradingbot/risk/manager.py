from dataclasses import dataclass

@dataclass
class Position:
    qty: float = 0.0

class RiskManager:
    def __init__(self, max_pos: float = 1.0):
        self.max_pos = max_pos
        self.pos = Position()

    def set_position(self, qty: float) -> None:
        """Sincroniza la posición mantenida por el RiskManager con el qty real."""
        self.pos.qty = float(qty)

    def add_fill(self, side: str, qty: float) -> None:
        """Actualiza posición interna tras un fill."""
        signed = float(qty) if side == "buy" else -float(qty)
        self.pos.qty += signed

    def size(self, signal_side: str, strength: float = 1.0) -> float:
        """
        Sizing trivial: objetivo ±max_pos, delta = objetivo - pos_actual.
        strength queda reservado para evolución futura (escalar tamaño).
        """
        target = self.max_pos * (1 if signal_side == "buy" else -1 if signal_side == "sell" else 0)
        return target - self.pos.qty
