from .breakout_atr import BreakoutATR
from .momentum import Momentum
from .mean_reversion import MeanReversion

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutATR.name: BreakoutATR,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
}

__all__ = ["BreakoutATR", "Momentum", "MeanReversion", "STRATEGIES"]
