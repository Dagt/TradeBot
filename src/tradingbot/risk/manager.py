from dataclasses import dataclass


@dataclass
class Position:
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
    ):
        self.max_pos = max_pos
        self.stop_loss_pct = abs(stop_loss_pct)
        self.max_drawdown_pct = abs(max_drawdown_pct)
        self.enabled = True
        self.pos = Position()
        # Variables internas para cálculo de SL/DD
        self._entry_price: float | None = None
        self._peak_price: float | None = None  # máximo a favor (long) / mínimo a favor (short)
        self._trough_price: float | None = None

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

    def size(self, signal_side: str, strength: float = 1.0) -> float:
        """
        Sizing trivial: objetivo ±max_pos, delta = objetivo - pos_actual.
        strength queda reservado para evolución futura (escalar tamaño).
        """
        target = self.max_pos * (1 if signal_side == "buy" else -1 if signal_side == "sell" else 0)
        return target - self.pos.qty

    def check_limits(self, price: float) -> bool:
        """Verifica stop loss y drawdown.

        Actualiza precios de referencia y retorna ``True`` si aún se pueden
        operar órdenes.  Si un límite se viola, ``enabled`` pasa a ``False`` y
        la función devuelve ``False``.
        """

        if not self.enabled:
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
            self.enabled = False
        if self.max_drawdown_pct > 0 and drawdown >= self.max_drawdown_pct:
            self.enabled = False

        return self.enabled
