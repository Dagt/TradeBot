"""Simple in-memory strategy state API."""

from fastapi import APIRouter

router = APIRouter()

_state: dict[str, str] = {}


@router.get("/strategies/status")
def strategies_status() -> dict:
    """Return the status of all strategies."""

    return {"strategies": _state}


@router.post("/strategies/{name}/{status}")
def set_strategy_status(name: str, status: str) -> dict:
    """Update the status of a strategy."""

    _state[name] = status
    return {"strategy": name, "status": status}


__all__ = ["router"]

