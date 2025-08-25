from .breakout_atr import BreakoutATR
from .breakout_vol import BreakoutVol
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .arbitrage_triangular import TriangularArb
from .cash_and_carry import CashAndCarry
from .order_flow import OrderFlow
from .depth_imbalance import DepthImbalance
from .liquidity_events import LiquidityEvents
from .triple_barrier import TripleBarrier
from .ml_models import MLStrategy
from .mean_rev_ofi import MeanRevOFI

# Registry of available strategies, useful for CLI/backtests
STRATEGIES = {
    BreakoutATR.name: BreakoutATR,
    BreakoutVol.name: BreakoutVol,
    Momentum.name: Momentum,
    MeanReversion.name: MeanReversion,
    TriangularArb.name: TriangularArb,
    CashAndCarry.name: CashAndCarry,
    OrderFlow.name: OrderFlow,
    DepthImbalance.name: DepthImbalance,
    LiquidityEvents.name: LiquidityEvents,
    TripleBarrier.name: TripleBarrier,
    MLStrategy.name: MLStrategy,
    MeanRevOFI.name: MeanRevOFI,
}

# Human-readable information about each strategy
STRATEGY_INFO = {
    "breakout_atr": {
        "desc": "Breakout based on ATR and Keltner channels",
        "requires": ["ohlcv"],
    },
    "breakout_vol": {
        "desc": "Volatility breakout using rolling standard deviation",
        "requires": ["ohlcv"],
    },
    "momentum": {
        "desc": "RSI-based momentum strategy",
        "requires": ["ohlcv"],
    },
    "mean_reversion": {
        "desc": "RSI mean reversion strategy",
        "requires": ["ohlcv"],
    },
    "triangular_arb": {
        "desc": "Naive triangular arbitrage across three markets",
        "requires": ["prices"],
    },
    "cash_and_carry": {
        "desc": "Cash and carry using spot, perp and funding basis",
        "requires": ["spot", "perp", "funding"],
    },
    "order_flow": {
        "desc": "Order flow imbalance from book deltas",
        "requires": ["bba", "book_delta"],
    },
    "depth_imbalance": {
        "desc": "Depth imbalance on top of the book",
        "requires": ["bba", "book_delta"],
    },
    "liquidity_events": {
        "desc": "Liquidity vacuum and gap events",
        "requires": ["bba", "book_delta"],
    },
    "triple_barrier": {
        "desc": "Triple-barrier labeling with boosting model",
        "requires": ["ohlcv"],
    },
    "ml": {
        "desc": "Machine learning driven strategy",
        "requires": ["features"],
    },
    "mean_rev_ofi": {
        "desc": "OFI z-score mean reversion under low volatility",
        "requires": ["ohlcv", "book_delta"],
    },
}

__all__ = [
    "BreakoutATR",
    "BreakoutVol",
    "Momentum",
    "MeanReversion",
    "TriangularArb",
    "CashAndCarry",
    "OrderFlow",
    "DepthImbalance",
    "LiquidityEvents",
    "TripleBarrier",
    "MLStrategy",
    "MeanRevOFI",
    "STRATEGIES",
    "STRATEGY_INFO",
]
