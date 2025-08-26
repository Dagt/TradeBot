import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from tradingbot.risk.manager import RiskManager, load_positions
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.risk.oco import OcoBook, load_active_oco


def test_rehydrate_state():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE "market.positions" (venue TEXT, symbol TEXT, qty REAL, avg_price REAL, realized_pnl REAL, fees_paid REAL);'))
        conn.execute(text('CREATE TABLE "market.oco_orders" (venue TEXT, symbol TEXT, side TEXT, qty REAL, entry_price REAL, sl_price REAL, tp_price REAL, status TEXT);'))
        conn.execute(text('INSERT INTO "market.positions" (venue, symbol, qty, avg_price, realized_pnl, fees_paid) VALUES ("paper", "BTCUSDT", 1.5, 10000, 0, 0);'))
        conn.execute(text('INSERT INTO "market.oco_orders" (venue, symbol, side, qty, entry_price, sl_price, tp_price, status) VALUES ("paper", "BTCUSDT", "long", 1.5, 10000, 9500, 10500, "active");'))

    rm = RiskManager()
    guard = PortfolioGuard(GuardConfig(total_cap_usdt=1e6, per_symbol_cap_usdt=1e6, venue="paper"))
    risk = RiskService(rm, guard)

    # Rehydrate
    pos_map = load_positions(engine, "paper")
    for sym, data in pos_map.items():
        risk.update_position("paper", sym, data["qty"])
    book = OcoBook()
    book.preload(load_active_oco(engine, venue="paper", symbols=["BTCUSDT"]))

    assert risk.rm.positions_multi["paper"]["BTCUSDT"] == pytest.approx(1.5)
    oco = book.get("BTCUSDT")
    assert oco is not None
    assert oco.sl_price == pytest.approx(9500.0)
