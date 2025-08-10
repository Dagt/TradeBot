from dataclasses import dataclass

@dataclass
class Position:
    qty: float = 0.0

class RiskManager:
    def __init__(self, max_pos: float = 1.0):
        self.max_pos = max_pos
        self.pos = Position()

    def size(self, signal_side: str, strength: float = 1.0) -> float:
        # Sizing trivial por ahora
        target = self.max_pos * (1 if signal_side == "buy" else -1 if signal_side == "sell" else 0)
        return target - self.pos.qty
