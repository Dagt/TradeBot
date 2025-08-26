from __future__ import annotations

from .manager import RiskManager
from .profile import RiskProfile


class EquityRiskManager(RiskManager):
    """Specialised :class:`RiskManager` driven by a :class:`RiskProfile`.

    The profile centralises percentage based parameters making it easier to
    tweak risk appetite from a single place.  Values missing from the profile
    fall back to those supplied directly via ``RiskManager``'s constructor.
    """

    def __init__(self, profile: RiskProfile, **kwargs):
        kwargs.setdefault("stop_loss_pct", profile.risk_per_trade_pct)
        kwargs.setdefault("max_drawdown_pct", profile.max_drawdown_pct)
        super().__init__(**kwargs)
        self.profile = profile
        # Scaling applies to volatility target/max position when used
        self.vol_target *= profile.scale_pct
        self.max_pos *= profile.scale_pct
        self._base_max_pos = self.max_pos
        self._base_vol_target = self.vol_target
