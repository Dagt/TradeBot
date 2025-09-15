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
    ORDERS,
    SKIPS,
    CANCELS,
    MAKER_TAKER_RATIO,
    KILL_SWITCH_ACTIVE,
    WS_FAILURES,
    TRADING_PNL,
    OPEN_POSITIONS,
    MARKET_LATENCY,
    E2E_LATENCY,
    SIGNAL_CONFIRM_LATENCY,
    FUNDING_RATE,
    BASIS,
    OPEN_INTEREST,
)

# System metrics
SYSTEM_DISCONNECTS = Counter(
    "system_disconnections_total",
    "Total number of system disconnections",
)

# CLI process completion metrics
CLI_PROCESS_COMPLETED = Counter(
    "cli_process_completed_total",
    "CLI processes completed successfully",
)

CLI_PROCESS_TIMEOUT = Counter(
    "cli_process_timeout_total",
    "CLI processes terminated due to timeout",
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
        abs(s.value) for s in slippage_samples if s.name.endswith("_sum")
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


def _histogram_percentile(metric, percentile: float) -> float:
    """Approximate a percentile for a prometheus histogram."""

    buckets: dict[float, float] = {}
    for metric_obj in metric.collect():
        for sample in metric_obj.samples:
            if sample.name.endswith("_bucket"):
                le = float(sample.labels.get("le", "inf"))
                buckets[le] = buckets.get(le, 0.0) + sample.value

    if not buckets:
        return 0.0

    total = max(buckets.values())
    target = percentile * total
    for le, count in sorted(buckets.items()):
        if count >= target:
            return le
    return list(sorted(buckets.items()))[-1][0]


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

    orders = ORDERS._value.get()
    cancels = CANCELS._value.get()
    skips = SKIPS._value.get()
    skip_rate = skips / orders if orders else 0.0

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
        "orders": orders,
        "cancels": cancels,
        "skips": skips,
        "skip_rate": skip_rate,
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
        "p90": _histogram_percentile(ORDER_LATENCY, 0.90),
        "p99": _histogram_percentile(ORDER_LATENCY, 0.99),
    }


@router.get("/metrics/positions")
def metrics_positions() -> dict:
    """Expose current positions with funding rates and basis."""

    return {
        "positions": _collect_by_symbol(OPEN_POSITIONS, "open_position"),
        "funding_rates": _collect_by_symbol(FUNDING_RATE, "funding_rate"),
        "basis": _collect_by_symbol(BASIS, "basis"),
    }


def _avg_slippage_by_symbol() -> dict[str, float]:
    """Compute average slippage per symbol from histogram samples."""

    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for metric_obj in SLIPPAGE.collect():
        for sample in metric_obj.samples:
            sym = sample.labels.get("symbol")
            if sample.name.endswith("_sum"):
                sums[sym] = sums.get(sym, 0.0) + abs(sample.value)
            elif sample.name.endswith("_count"):
                counts[sym] = counts.get(sym, 0.0) + sample.value
    return {sym: (sums.get(sym, 0.0) / counts.get(sym, 1.0)) for sym in sums}


@router.get("/metrics/execution")
def metrics_execution() -> dict:
    """Expose basic execution quality stats."""

    slips = _avg_slippage_by_symbol()
    items = [
        {"symbol": sym, "is": 0.0, "slippage": val, "fill_ratio": 0.0}
        for sym, val in slips.items()
    ]
    return {"items": items}


@router.get("/metrics/signals")
def metrics_signals() -> dict:
    """Expose signal quality metrics (placeholder)."""

    return {"items": []}


@router.get("/metrics/risk")
def metrics_risk() -> dict:
    """Expose counts of risk management events."""

    triggers: dict[str, float] = {}
    for metric_obj in RISK_EVENTS.collect():
        for sample in metric_obj.samples:
            if sample.name.endswith("_total"):
                evt = sample.labels.get("event_type", "unknown")
                triggers[evt] = triggers.get(evt, 0.0) + sample.value
    return {"triggers": triggers}


@router.get("/metrics/market")
def metrics_market() -> dict:
    """Expose basic market state information."""

    basis_vals = _collect_by_symbol(BASIS, "basis")
    avg_basis = (
        sum(basis_vals.values()) / len(basis_vals) if basis_vals else 0.0
    )
    funding_vals = _collect_by_symbol(FUNDING_RATE, "funding_rate")
    avg_funding = (
        sum(funding_vals.values()) / len(funding_vals) if funding_vals else 0.0
    )
    return {
        "spread": 0.0,
        "depth": 0.0,
        "funding_current": avg_funding,
        "funding_next": 0.0,
        "basis": avg_basis,
        "basis_alert": abs(avg_basis) > 0.05,
    }
