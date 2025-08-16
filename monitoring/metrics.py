from fastapi import APIRouter, Response
from prometheus_client import Gauge, Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST

# Trading metrics
TRADING_PNL = Gauge(
    "trading_pnl",
    "Current trading profit and loss in USD",
)

# Market metrics
MARKET_LATENCY = Histogram(
    "market_latency_seconds",
    "Latency of market data processing in seconds",
)

# System metrics
SYSTEM_DISCONNECTS = Counter(
    "system_disconnections_total",
    "Total number of system disconnections",
)

router = APIRouter()


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/summary")
def metrics_summary() -> dict:
    """Return a minimal summary of key metrics."""
    return {
        "pnl": TRADING_PNL._value.get(),
        "disconnects": SYSTEM_DISCONNECTS._value.get(),
    }
