from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Reuse detailed execution metrics
from tradingbot.utils.metrics import (
    FILL_COUNT,
    SLIPPAGE,
    RISK_EVENTS,
    ORDER_LATENCY,
    MAKER_TAKER_RATIO,
    WS_FAILURES,
)

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

# End-to-end metrics
E2E_LATENCY = Histogram(
    "e2e_latency_seconds",
    "End-to-end latency of the trading pipeline in seconds",
    labelnames=["strategy"],
)

STRATEGY_UP = Gauge(
    "strategy_up",
    "Whether a trading strategy is running (1) or not (0)",
    labelnames=["strategy"],
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

    # Compute average order execution latency across venues
    latency_samples = [
        sample
        for metric in ORDER_LATENCY.collect()
        for sample in metric.samples
    ]
    latency_sum = sum(s.value for s in latency_samples if s.name.endswith("_sum"))
    latency_count = sum(
        s.value for s in latency_samples if s.name.endswith("_count")
    )
    avg_latency = latency_sum / latency_count if latency_count else 0.0

    # Compute average maker/taker ratio across venues
    ratio_samples = [
        sample.value
        for metric in MAKER_TAKER_RATIO.collect()
        for sample in metric.samples
        if sample.name == "maker_taker_ratio"
    ]
    avg_ratio = sum(ratio_samples) / len(ratio_samples) if ratio_samples else 0.0

    # Aggregate websocket failures across adapters
    ws_total = sum(
        sample.value
        for metric in WS_FAILURES.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )

    # Compute average end-to-end latency across strategies
    e2e_samples = [
        sample
        for metric in E2E_LATENCY.collect()
        for sample in metric.samples
    ]
    e2e_sum = sum(s.value for s in e2e_samples if s.name.endswith("_sum"))
    e2e_count = sum(s.value for s in e2e_samples if s.name.endswith("_count"))
    avg_e2e_latency = e2e_sum / e2e_count if e2e_count else 0.0

    # Capture strategy status
    strategy_status = {
        sample.labels.get("strategy"): sample.value
        for metric in STRATEGY_UP.collect()
        for sample in metric.samples
        if sample.name == "strategy_up"
    }

    return {
        "pnl": TRADING_PNL._value.get(),
        "disconnects": SYSTEM_DISCONNECTS._value.get(),
        "fills": fill_total,
        "risk_events": risk_total,
        "avg_slippage_bps": avg_slippage,
        "avg_order_latency_seconds": avg_latency,
        "avg_maker_taker_ratio": avg_ratio,
        "ws_failures": ws_total,
        "avg_e2e_latency_seconds": avg_e2e_latency,
        "strategies": strategy_status,
    }
