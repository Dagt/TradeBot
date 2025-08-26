from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict

import yaml


@dataclass
class RiskProfile:
    """Configuration container for percentage based risk parameters.

    Attributes
    ----------
    risk_per_trade_pct:
        Fraction of equity to risk on a single trade (e.g. ``0.02`` for 2%).
    max_drawdown_pct:
        Maximum allowed drawdown expressed as fraction of peak equity.
    scale_pct:
        Multiplicative factor applied to scalable limits (``1.0`` leaves
        original values untouched).
    """

    risk_per_trade_pct: float = 0.01
    max_drawdown_pct: float = 0.2
    scale_pct: float = 1.0

    # ------------------------------------------------------------------
    # serialisation helpers
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskProfile":
        return cls(
            risk_per_trade_pct=float(data.get("risk_per_trade_pct", 0.01)),
            max_drawdown_pct=float(data.get("max_drawdown_pct", 0.2)),
            scale_pct=float(data.get("scale_pct", 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # YAML --------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RiskProfile":
        with open(path, "r", encoding="utf8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf8") as fh:
            yaml.safe_dump(self.to_dict(), fh)

    # JSON --------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> "RiskProfile":
        with open(path, "r", encoding="utf8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.to_dict(), fh, indent=indent)
