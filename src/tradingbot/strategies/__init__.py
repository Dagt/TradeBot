from .breakout_atr import BreakoutATR
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .order_flow import OrderFlow

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutATR.name: BreakoutATR,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
    OrderFlow.name: OrderFlow,
}

__all__ = ["BreakoutATR", "Momentum", "MeanReversion", "OrderFlow", "STRATEGIES"]
