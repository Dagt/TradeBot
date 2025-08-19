"""Simple in-memory strategy state and configuration API."""

from fastapi import APIRouter, HTTPException

from .metrics import STRATEGY_STATE

router = APIRouter()

# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------
# ``_available`` tracks all strategies known to the monitoring panel.  ``_state``
# stores the current execution status for each strategy while ``_params`` holds
# arbitrary configuration parameters supplied by the user.  Everything is kept
# in-memory and is therefore ephemeral but sufficient for the lightweight panel
# used in tests and local setups.

_available: set[str] = set()
_state: dict[str, str] = {}
_params: dict[str, dict] = {}

_STATE_MAP = {"stopped": 0, "running": 1, "error": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def register_strategy(name: str, params: dict | None = None) -> None:
    """Register a new strategy.

    Parameters are optional and can be provided later via
    :func:`update_strategy_params`.
    """

    _available.add(name)
    _state.setdefault(name, "stopped")
    if params:
        _params[name] = params


def available_strategies() -> dict:
    """Return the list of registered strategy names."""

    return {"strategies": sorted(_available)}


def update_strategy_params(name: str, params: dict) -> dict:
    """Persist parameters for ``name`` and return them."""

    if name not in _available:
        raise HTTPException(status_code=404, detail="Strategy not registered")
    _params[name] = params
    return {"strategy": name, "params": params}


def get_strategy_params(name: str) -> dict:
    """Return stored parameters for ``name``."""

    if name not in _available:
        raise HTTPException(status_code=404, detail="Strategy not registered")
    return {"strategy": name, "params": _params.get(name, {})}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/strategies")
def strategies() -> dict:
    """Expose the names of all registered strategies."""

    return available_strategies()


@router.get("/strategies/status")
def strategies_status() -> dict:
    """Return the status and stored parameters of all strategies."""

    return {
        "strategies": {
            name: {"status": _state.get(name, "unknown"), "params": _params.get(name, {})}
            for name in _available
        }
    }


@router.post("/strategies/{name}/{status}")
def set_strategy_status(name: str, status: str) -> dict:
    """Update the status of a strategy."""

    if name not in _available:
        _available.add(name)
    _state[name] = status
    metric_value = _STATE_MAP.get(status.lower(), -1)
    STRATEGY_STATE.labels(strategy=name).set(metric_value)
    return {"strategy": name, "status": status}


@router.post("/strategies/{name}/params")
def set_params_endpoint(name: str, params: dict) -> dict:
    """Endpoint wrapper for :func:`update_strategy_params`."""

    return update_strategy_params(name, params)


@router.get("/strategies/{name}/params")
def get_params_endpoint(name: str) -> dict:
    """Return parameters for ``name``."""

    return get_strategy_params(name)


__all__ = [
    "router",
    "register_strategy",
    "available_strategies",
    "update_strategy_params",
    "get_strategy_params",
    "strategies_status",
    "set_strategy_status",
]

