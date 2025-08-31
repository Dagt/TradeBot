import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from tradingbot.core import Account
from tradingbot.risk.manager import load_positions
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


def test_rehydrate_state():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE "market.positions" (venue TEXT, symbol TEXT, qty REAL, avg_price REAL, realized_pnl REAL, fees_paid REAL);'))
        conn.execute(text('INSERT INTO "market.positions" (venue, symbol, qty, avg_price, realized_pnl, fees_paid) VALUES ("paper", "BTCUSDT", 1.5, 10000, 0, 0);'))

    account = Account(float("inf"))
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="paper"))
    guard.refresh_usd_caps(1e6)
    risk = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.0,
    )

    # Rehydrate
    pos_map = load_positions(engine, "paper")
    for sym, data in pos_map.items():
        risk.update_position("paper", sym, data["qty"])
    assert risk.rm.positions_multi["paper"]["BTCUSDT"] == pytest.approx(1.5)
