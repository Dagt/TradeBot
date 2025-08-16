from .breakout_atr import BreakoutATR
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .cash_and_carry import CashAndCarry

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutATR.name: BreakoutATR,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
    CashAndCarry.name: CashAndCarry,
}

__all__ = ["BreakoutATR", "Momentum", "MeanReversion", "CashAndCarry", "STRATEGIES"]
