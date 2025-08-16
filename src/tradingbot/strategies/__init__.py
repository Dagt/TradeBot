from .breakout_atr import BreakoutATR
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .cash_and_carry import CashAndCarry
from .order_flow import OrderFlow

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutATR.name: BreakoutATR,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
    CashAndCarry.name: CashAndCarry,
    OrderFlow.name: OrderFlow,
}

__all__ = ["BreakoutATR", "Momentum", "MeanReversion", "CashAndCarry", "OrderFlow", "STRATEGIES"]
