import pytest
from sqlalchemy import create_engine, text

from tradingbot.risk.manager import RiskManager, rehydrate_positions
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService
from tradingbot.risk.oco import OcoBook, load_open_oco
from tradingbot.storage import timescale


def _setup_engine():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS market")
        conn.execute(
            text(
                """
                CREATE TABLE market.positions (
                    venue TEXT,
                    symbol TEXT,
                    qty REAL,
                    avg_price REAL,
                    realized_pnl REAL,
                    fees_paid REAL,
                    UNIQUE (venue, symbol)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE market.oco_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    entry_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    status TEXT,
                    triggered TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
        )
    return engine


def test_rehydrate_positions_and_oco():
    engine = _setup_engine()
    timescale.upsert_position(
        engine,
        venue="binance",
        symbol="BTCUSDT",
        qty=1.5,
        avg_price=100.0,
        realized_pnl=0.0,
        fees_paid=0.0,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO market.oco_orders (
                    venue, symbol, side, qty, entry_price, sl_price, tp_price, status
                ) VALUES (:v, :s, :side, :qty, :entry, :sl, :tp, 'active')
                """
            ),
            {
                "v": "binance",
                "s": "BTCUSDT",
                "side": "long",
                "qty": 1.5,
                "entry": 100.0,
                "sl": 90.0,
                "tp": 110.0,
            },
        )

    rm = RiskManager(max_pos=5.0)
    guard = PortfolioGuard(GuardConfig(venue="binance"))
    risk = RiskService(rm, guard)
    oco_book = OcoBook()

    positions = rehydrate_positions(engine, "binance", risk)
    oco_book.preload(load_open_oco(engine, "binance", ["BTCUSDT"]))

    assert positions == {"BTCUSDT": pytest.approx(1.5)}
    assert rm.positions_multi["binance"]["BTCUSDT"] == pytest.approx(1.5)
    oco = oco_book.get("BTCUSDT")
    assert oco is not None
    assert oco.sl_price == pytest.approx(90.0)
    assert oco.tp_price == pytest.approx(110.0)
