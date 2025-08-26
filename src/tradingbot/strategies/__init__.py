from .breakout_atr import BreakoutATR
from .breakout_vol import BreakoutVol
from .momentum import Momentum
from .mean_reversion import MeanReversion
from .trend_following import TrendFollowing
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
    TrendFollowing.name: TrendFollowing,
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

# Metadata describing each strategy.  ``desc`` provides a short human
# readable description while ``requires`` lists the data inputs needed by the
# strategy when running in real time or backtests.  This information is used by
# the API and front-end to display helpful details to the user.
STRATEGY_INFO: dict[str, dict] = {
    "breakout_atr": {
        "desc": "Ruptura basada en canales ATR",
        "requires": ["ohlcv"],
    },
    "breakout_vol": {
        "desc": "Ruptura por volatilidad",
        "requires": ["ohlcv"],
    },
    "momentum": {
        "desc": "Sigue la tendencia con RSI",
        "requires": ["ohlcv"],
    },
    "trend_following": {
        "desc": "Seguimiento de tendencia con fuerza adaptativa",
        "requires": ["ohlcv"],
    },
    "mean_reversion": {
        "desc": "Reversión a la media con RSI",
        "requires": ["ohlcv"],
    },
    "triangular_arb": {
        "desc": "Arbitraje entre tres pares",
        "requires": ["prices"],
    },
    "cash_and_carry": {
        "desc": "Diferencias entre spot y perp con funding",
        "requires": ["spot", "perp", "funding"],
    },
    "order_flow": {
        "desc": "Desequilibrio del flujo de órdenes",
        "requires": ["bba", "book_delta"],
    },
    "depth_imbalance": {
        "desc": "Desequilibrio de profundidad del libro",
        "requires": ["book_delta"],
    },
    "liquidity_events": {
        "desc": "Detecta vacíos y gaps de liquidez",
        "requires": ["bba", "book_delta"],
    },
    "triple_barrier": {
        "desc": "Señales con etiquetado de triple barrera",
        "requires": ["ohlcv"],
    },
    "ml": {
        "desc": "Modelo de machine learning",
        "requires": ["features"],
    },
    "mean_rev_ofi": {
        "desc": "Reversión a la media con OFI",
        "requires": ["book_delta"],
    },
}

__all__ = [
    "BreakoutATR",
    "BreakoutVol",
    "Momentum",
    "MeanReversion",
    "TrendFollowing",
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
