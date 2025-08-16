from fastapi import APIRouter, Response
from prometheus_client import Gauge, Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST

# Reuse detailed execution metrics
from tradingbot.utils.metrics import FILL_COUNT, SLIPPAGE, RISK_EVENTS

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

    # Aggregate fills across all symbols/sides
    fill_total = sum(
        sample.value
        for metric in FILL_COUNT.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )

    # Aggregate risk events across all event types
    risk_total = sum(
        sample.value
        for metric in RISK_EVENTS.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )

    # Compute average slippage in basis points
    slippage_samples = [
        sample
        for metric in SLIPPAGE.collect()
        for sample in metric.samples
    ]
    slippage_sum = sum(s.value for s in slippage_samples if s.name.endswith("_sum"))
    slippage_count = sum(
        s.value for s in slippage_samples if s.name.endswith("_count")
    )
    avg_slippage = slippage_sum / slippage_count if slippage_count else 0.0

    return {
        "pnl": TRADING_PNL._value.get(),
        "disconnects": SYSTEM_DISCONNECTS._value.get(),
        "fills": fill_total,
        "risk_events": risk_total,
        "avg_slippage_bps": avg_slippage,
    }
