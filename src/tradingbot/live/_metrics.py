from __future__ import annotations

"""Helpers for enriching live trading metrics."""

from collections.abc import Mapping
from typing import Any


_TRUE_STRINGS = {"true", "1", "yes", "y", "maker"}
_FALSE_STRINGS = {"false", "0", "no", "n", "taker"}


def _coerce_bool(value: Any) -> bool | None:
    """Best-effort coercion of heterogeneous truthy values to ``bool``.

    The values returned by different adapters can be integers, strings or even
    explicit booleans.  This helper normalises them so callers can rely on a
    proper ``True``/``False`` instead of re-implementing the conversion.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        return None
    try:
        return bool(value)
    except Exception:  # pragma: no cover - defensive
        return None


def infer_maker_flag(
    res: Mapping[str, Any] | None,
    exec_price: float | None,
    base_price: Any,
) -> bool:
    """Infer whether a fill acted as maker or taker.

    Parameters
    ----------
    res:
        Raw execution payload returned by the adapter.
    exec_price:
        Price at which the fill executed.
    base_price:
        Original limit price requested by the order.
    """

    payload: Mapping[str, Any]
    if isinstance(res, Mapping):
        payload = res
    else:
        payload = {}

    maker_flag = None

    maker_raw = payload.get("maker")
    if maker_raw is not None:
        maker_flag = _coerce_bool(maker_raw)

    if maker_flag is None:
        fee_type = payload.get("fee_type")
        if isinstance(fee_type, str):
            fee_type_norm = fee_type.strip().lower()
            if fee_type_norm == "maker":
                maker_flag = True
            elif fee_type_norm == "taker":
                maker_flag = False

    if maker_flag is None and exec_price is not None:
        base_candidate = base_price
        if base_candidate is None:
            base_candidate = payload.get("price") or payload.get("avg_price")
        try:
            if base_candidate is not None:
                base_val = float(base_candidate)
            else:
                base_val = None
        except (TypeError, ValueError):
            base_val = None

        if base_val is not None:
            tolerance = max(abs(base_val), abs(exec_price)) * 1e-9
            tolerance = max(tolerance, 1e-9)
            if abs(exec_price - base_val) <= tolerance:
                maker_flag = True
            else:
                maker_flag = False

    return bool(maker_flag)
