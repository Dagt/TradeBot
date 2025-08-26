from tradingbot.risk import (
    RiskProfile,
    EquityRiskManager,
    PortfolioGuard,
    DailyGuard,
    GuardLimits,
)
from tradingbot.risk.portfolio_guard import GuardConfig


def test_profile_serialisation(tmp_path):
    profile = RiskProfile(risk_per_trade_pct=0.02, max_drawdown_pct=0.1, scale_pct=1.5)
    yfile = tmp_path / "profile.yaml"
    jfile = tmp_path / "profile.json"

    profile.to_yaml(yfile)
    profile.to_json(jfile)

    assert RiskProfile.from_yaml(yfile) == profile
    assert RiskProfile.from_json(jfile) == profile


def test_components_read_profile():
    profile = RiskProfile(risk_per_trade_pct=0.03, max_drawdown_pct=0.07, scale_pct=0.5)

    rm = EquityRiskManager(profile)
    assert rm.stop_loss_pct == profile.risk_per_trade_pct
    assert rm.max_drawdown_pct == profile.max_drawdown_pct
    assert rm.max_pos == 1.0 * profile.scale_pct

    pg_cfg = GuardConfig()
    pg = PortfolioGuard(pg_cfg, profile=profile)
    assert pg.cfg.soft_cap_pct == profile.scale_pct

    dg = DailyGuard(GuardLimits(), venue="paper", profile=profile)
    assert dg.lim.daily_max_drawdown_pct == profile.max_drawdown_pct
