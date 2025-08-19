"""Status server exposing strategy states and basic metrics."""
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from typing import Dict

from ..strategies import STRATEGIES
from ..utils.metrics import REQUEST_COUNT, WS_FAILURES
from ..logging_conf import setup_logging

# Configure logging and sentry if available
setup_logging()

app = FastAPI(title="TradingBot Status")

# In-memory store for strategy states; external modules may update.
_STRATEGY_STATES: Dict[str, str] = {name: "idle" for name in STRATEGIES.keys()}


@app.get("/health")
def health() -> dict:
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.get("/strategies")
def strategies() -> dict:
    """Return the current state of all known strategies."""
    return {
        "strategies": [
            {"name": name, "state": state} for name, state in _STRATEGY_STATES.items()
        ]
    }


@app.post("/strategies/{name}/{state}")
def set_strategy_state(name: str, state: str) -> dict:
    """Update the state of a given strategy.

    This is a simple in-memory setter useful for demos or local monitoring.
    """
    if name not in _STRATEGY_STATES:
        return {"error": "unknown strategy"}
    _STRATEGY_STATES[name] = state
    return {"name": name, "state": state}


def _counter_value(counter) -> float:
    """Extract the value of a Prometheus counter."""
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name == counter._name and not sample.labels:
                return float(sample.value)
    return 0.0


def metrics_summary() -> dict:
    """Return a minimal summary of key metrics."""
    return {
        "request_count": _counter_value(REQUEST_COUNT),
        "ws_failures": _counter_value(WS_FAILURES),
    }


@app.get("/metrics")
def metrics() -> dict:
    """Expose aggregated metrics in JSON format."""
    return metrics_summary()


@app.get("/metrics/prometheus")
def metrics_prometheus() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
