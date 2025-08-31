import pandas as pd
import pytest

from tradingbot.core import Account
from tradingbot.data.features import book_vacuum, liquidity_gap
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
    account = Account(float("inf"))
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = LiquidityEvents(
        vacuum_threshold=0.5,
        gap_threshold=2,
        dynamic_thresholds=False,
        risk_service=svc,
    )
    sig = strat.on_bar({"window": df_entry})
    assert sig is not None and sig.side == "buy"
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy")
    assert trade["stop"] == pytest.approx(expected_stop)


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
