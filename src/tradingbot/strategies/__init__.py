"""Trading strategies and registry.

Each strategy can be instantiated dynamically via the :data:`STRATEGIES`
registry::

    from tradingbot.strategies import STRATEGIES
    strat = STRATEGIES["momentum"](rsi_n=14)

The classes accept parameters using ``**kwargs`` to ease construction from
configuration files or command line arguments.
"""

from .breakout_vol import BreakoutVol
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .triangular_arb import TriangularArb
from .cash_and_carry import CashAndCarry
from .order_flow import OrderFlow

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutVol.name: BreakoutVol,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
    TriangularArb.name: TriangularArb,
    CashAndCarry.name: CashAndCarry,
    OrderFlow.name: OrderFlow,
}

__all__ = [
    "BreakoutVol",
    "Momentum",
    "MeanReversion",
    "TriangularArb",
    "CashAndCarry",
    "OrderFlow",
    "STRATEGIES",
]
