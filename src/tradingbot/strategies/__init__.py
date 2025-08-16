from .breakout_atr import BreakoutATR
from .breakout_vol import BreakoutVol
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .arbitrage_triangular import TriangularArb
from .cash_and_carry import CashAndCarry
from .order_flow import OrderFlow
from .triple_barrier import TripleBarrier

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutATR.name: BreakoutATR,
    BreakoutVol.name: BreakoutVol,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
    TriangularArb.name: TriangularArb,
    CashAndCarry.name: CashAndCarry,
    OrderFlow.name: OrderFlow,
    TripleBarrier.name: TripleBarrier,
}

__all__ = [
    "BreakoutATR",
    "BreakoutVol",
    "Momentum",
    "MeanReversion",
    "TriangularArb",
    "CashAndCarry",
    "OrderFlow",
    "TripleBarrier",
    "STRATEGIES",
]
