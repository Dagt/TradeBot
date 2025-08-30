import pandas as pd
import pytest

from tradingbot.data.features import book_vacuum, liquidity_gap
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.strategies.liquidity_events import LiquidityEvents


def test_book_vacuum_detection():
    df = pd.DataFrame({"bid_qty": [10, 10], "ask_qty": [10, 4]})
    events = book_vacuum(df, threshold=0.5)
    assert list(events) == [0, 1]


def test_liquidity_gap_detection():
    df = pd.DataFrame({
        "bid_px": [[100, 99], [100, 95]],
        "ask_px": [[101, 102], [101, 102]],
    })
    events = liquidity_gap(df, threshold=2)
    assert list(events) == [0, -1]


def test_liquidity_events_strategy_buy_vacuum():
    df = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 4],
        "bid_px": [[100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df})
    assert sig is not None and sig.side == "buy"


def test_liquidity_events_strategy_sell_gap():
    df = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 10],
        "bid_px": [[100, 99], [100, 90]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=5, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df})
    assert sig is not None and sig.side == "sell"


def test_liquidity_events_no_signal_returns_none():
    df = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 10],
        "bid_px": [[100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df})
    assert sig is None


def test_liquidity_events_risk_service_handles_stop_and_size():
    df_entry = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 4],
        "bid_px": [[100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df_entry})
    assert sig is not None and sig.side == "buy"
    price = strat.entry_price

    rm = RiskManager(risk_pct=0.02)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(rm, guard)
    svc.account.update_cash(1000.0)

    allowed, reason, delta = svc.check_order(
        "AAA", sig.side, 1000.0, price, strength=sig.strength
    )
    assert allowed and delta > 0

    rm.add_fill(sig.side, delta, price=price)
    svc.update_position("X", "AAA", delta)

    stop_price = price * (1 - rm.risk_pct) - 1
    allowed, reason, delta_stop = svc.check_order(
        "AAA", sig.side, 1000.0, stop_price
    )
    assert allowed and reason == "stop_loss"
    assert delta_stop == pytest.approx(-delta)


def test_dynamic_thresholds_increase_events():
    df = pd.DataFrame({
        "bid_qty": [10, 10, 10, 10],
        "ask_qty": [10, 10, 10, 10],
        "bid_px": [[100, 99], [200, 199], [50, 49], [50, 48.5]],
        "ask_px": [[101, 102], [201, 202], [51, 52], [51, 52]],
    })
    strat_dyn = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, vol_window=3, dynamic_thresholds=True)
    sig_dyn = strat_dyn.on_bar({"window": df})
    assert sig_dyn is not None and sig_dyn.side == "sell"

    strat_static = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, dynamic_thresholds=False)
    sig_static = strat_static.on_bar({"window": df})
    assert sig_static is None
