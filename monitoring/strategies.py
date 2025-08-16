"""Simple in-memory strategy state API."""

from fastapi import APIRouter

from .metrics import STRATEGY_STATE

router = APIRouter()

_state: dict[str, str] = {}

_STATE_MAP = {"stopped": 0, "running": 1, "error": 2}


@router.get("/strategies/status")
def strategies_status() -> dict:
    """Return the status of all strategies."""

    return {"strategies": _state}


@router.post("/strategies/{name}/{status}")
def set_strategy_status(name: str, status: str) -> dict:
    """Update the status of a strategy."""

    _state[name] = status
    metric_value = _STATE_MAP.get(status.lower(), -1)
    STRATEGY_STATE.labels(strategy=name).set(metric_value)
    return {"strategy": name, "status": status}


__all__ = ["router"]

