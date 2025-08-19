from fastapi import APIRouter, Response
from prometheus_client import (
    Gauge,
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import psutil
import time

# Reuse detailed execution metrics
from tradingbot.utils.metrics import (
    FILL_COUNT,
    SLIPPAGE,
    RISK_EVENTS,
    ORDER_LATENCY,
    ORDER_SENT,
    ORDER_REJECTS,
    MAKER_TAKER_RATIO,
    KILL_SWITCH_ACTIVE,
    WS_FAILURES,
    TRADING_PNL,
    OPEN_POSITIONS,
    MARKET_LATENCY,
    E2E_LATENCY,
    FUNDING_RATE,
    BASIS,
    OPEN_INTEREST,
)

# System metrics
SYSTEM_DISCONNECTS = Counter(
    "system_disconnections_total",
    "Total number of system disconnections",
)

# Process gauges
PROCESS_CPU = Gauge(
    "process_cpu_percent",
    "Process CPU usage percent",
)
PROCESS_MEMORY = Gauge(
    "process_memory_bytes",
    "Process memory usage in bytes",
)
PROCESS_UPTIME = Gauge(
    "process_uptime_seconds",
    "Process uptime in seconds",
)

_process = psutil.Process()


def update_process_metrics() -> None:
    """Update process-level system metrics."""
    with _process.oneshot():
        PROCESS_CPU.set(_process.cpu_percent(interval=None))
        PROCESS_MEMORY.set(_process.memory_info().rss)
        PROCESS_UPTIME.set(time.time() - _process.create_time())


STRATEGY_STATE = Gauge(
    "strategy_state",
    "Current state of strategies (0=stopped,1=running,2=error)",
    ["strategy"],
)

STRATEGY_ACTIONS = Counter(
    "strategy_actions_total",
    "Total number of strategy control actions",
    ["strategy", "action"],
)

router = APIRouter()


def _collect_by_symbol(metric, sample_name: str) -> dict[str, float]:
    """Collect gauge samples keyed by symbol."""

    return {
        sample.labels["symbol"]: sample.value
        for metric_obj in metric.collect()
        for sample in metric_obj.samples
        if sample.name == sample_name
    }


def _avg_slippage() -> float:
    """Compute average slippage in basis points."""

    slippage_samples = [
        sample
        for metric in SLIPPAGE.collect()
        for sample in metric.samples
    ]
    slippage_sum = sum(
        s.value for s in slippage_samples if s.name.endswith("_sum")
    )
    slippage_count = sum(
        s.value for s in slippage_samples if s.name.endswith("_count")
    )
    return slippage_sum / slippage_count if slippage_count else 0.0


def _avg_order_latency() -> float:
    """Compute average order execution latency across venues."""

    latency_samples = [
        sample
        for metric in ORDER_LATENCY.collect()
        for sample in metric.samples
    ]
    latency_sum = sum(
        s.value for s in latency_samples if s.name.endswith("_sum")
    )
    latency_count = sum(
        s.value for s in latency_samples if s.name.endswith("_count")
    )
    return latency_sum / latency_count if latency_count else 0.0


def _avg_market_latency() -> float:
    """Compute average market data latency."""

    market_samples = [
        sample
        for metric in MARKET_LATENCY.collect()
        for sample in metric.samples
    ]
    market_sum = sum(
        s.value for s in market_samples if s.name.endswith("_sum")
    )
    market_count = sum(
        s.value for s in market_samples if s.name.endswith("_count")
    )
    return market_sum / market_count if market_count else 0.0


def _avg_e2e_latency() -> float:
    """Compute average end-to-end latency."""

    e2e_samples = [
        sample
        for metric in E2E_LATENCY.collect()
        for sample in metric.samples
    ]
    e2e_sum = sum(s.value for s in e2e_samples if s.name.endswith("_sum"))
    e2e_count = sum(
        s.value for s in e2e_samples if s.name.endswith("_count")
    )
    return e2e_sum / e2e_count if e2e_count else 0.0


def metrics_summary() -> dict:
    """Return a minimal summary of key metrics."""

    update_process_metrics()

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

    avg_slippage = _avg_slippage()
    avg_latency = _avg_order_latency()
    avg_market_latency = _avg_market_latency()
    avg_e2e = _avg_e2e_latency()

    orders_sent = ORDER_SENT._value.get()
    order_rejects = ORDER_REJECTS._value.get()
    reject_rate = order_rejects / orders_sent if orders_sent else 0.0

    # Compute average maker/taker ratio across venues
    ratio_samples = [
        sample.value
        for metric in MAKER_TAKER_RATIO.collect()
        for sample in metric.samples
        if sample.name == "maker_taker_ratio"
    ]
    avg_ratio = (
        sum(ratio_samples) / len(ratio_samples) if ratio_samples else 0.0
    )

    # Aggregate websocket failures across adapters
    ws_failures_total = sum(
        sample.value
        for metric in WS_FAILURES.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )

    # Capture current strategy states
    strategy_states = {
        sample.labels["strategy"]: sample.value
        for metric in STRATEGY_STATE.collect()
        for sample in metric.samples
        if sample.name == "strategy_state"
    }

    positions = _collect_by_symbol(OPEN_POSITIONS, "open_position")

    funding_rates: dict[str, float] = _collect_by_symbol(
        FUNDING_RATE, "funding_rate"
    )

    open_interest: dict[str, float] = _collect_by_symbol(
        OPEN_INTEREST, "open_interest"
    )

    basis = _collect_by_symbol(BASIS, "basis")

    return {
        "pnl": TRADING_PNL._value.get(),
        "positions": positions,
        "funding_rates": funding_rates,
        "open_interest": open_interest,
        "basis": basis,
        "disconnects": SYSTEM_DISCONNECTS._value.get(),
        "fills": fill_total,
        "risk_events": risk_total,
        "orders_sent": orders_sent,
        "order_rejects": order_rejects,
        "order_reject_rate": reject_rate,
        "kill_switch_active": KILL_SWITCH_ACTIVE._value.get(),
        "avg_slippage_bps": avg_slippage,
        "avg_market_latency_seconds": avg_market_latency,
        "avg_order_latency_seconds": avg_latency,
        "avg_maker_taker_ratio": avg_ratio,
        "avg_e2e_latency_seconds": avg_e2e,
        "ws_failures": ws_failures_total,
        "strategy_states": strategy_states,
        "cpu_percent": PROCESS_CPU._value.get(),
        "memory_bytes": PROCESS_MEMORY._value.get(),
        "process_uptime_seconds": PROCESS_UPTIME._value.get(),
    }


@router.get("/metrics")
def metrics() -> dict:
    """Expose aggregated metrics in JSON format."""

    return metrics_summary()


@router.get("/metrics/summary")
def metrics_summary_route() -> dict:
    """Expose a minimal summary of key metrics."""

    return metrics_summary()


@router.get("/metrics/prometheus")
def metrics_prometheus() -> Response:
    """Expose Prometheus metrics for scraping."""
    update_process_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/pnl")
def metrics_pnl() -> dict:
    """Expose current trading PnL."""

    return {"pnl": TRADING_PNL._value.get()}


@router.get("/metrics/slippage")
def metrics_slippage() -> dict:
    """Expose aggregated slippage information."""

    return {"avg_slippage_bps": _avg_slippage()}


@router.get("/metrics/latency")
def metrics_latency() -> dict:
    """Expose order and market latency metrics."""

    return {
        "avg_order_latency_seconds": _avg_order_latency(),
        "avg_market_latency_seconds": _avg_market_latency(),
        "avg_e2e_latency_seconds": _avg_e2e_latency(),
    }


@router.get("/metrics/positions")
def metrics_positions() -> dict:
    """Expose current positions with funding rates and basis."""

    return {
        "positions": _collect_by_symbol(OPEN_POSITIONS, "open_position"),
        "funding_rates": _collect_by_symbol(FUNDING_RATE, "funding_rate"),
        "basis": _collect_by_symbol(BASIS, "basis"),
    }
