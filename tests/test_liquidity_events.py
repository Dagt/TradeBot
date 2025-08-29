import pandas as pd

from tradingbot.data.features import book_vacuum, liquidity_gap
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


def test_take_profit_exit():
    df_entry = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 4],
        "bid_px": [[100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, tp_pct=0.01, sl_pct=0.01, max_hold=10, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df_entry})
    assert sig is not None and sig.side == "buy"

    df_exit = pd.DataFrame({
        "bid_qty": [10, 10, 10],
        "ask_qty": [10, 4, 10],
        "bid_px": [[100, 99], [100, 99], [102, 101]],
        "ask_px": [[101, 102], [101, 102], [103, 104]],
    })
    sig_exit = strat.on_bar({"window": df_exit})
    assert sig_exit is not None and sig_exit.side == "sell" and sig_exit.reduce_only


def test_stop_loss_exit():
    df_entry = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 4],
        "bid_px": [[100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, tp_pct=0.05, sl_pct=0.01, max_hold=10, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df_entry})
    assert sig is not None and sig.side == "buy"

    df_exit = pd.DataFrame({
        "bid_qty": [10, 10, 10],
        "ask_qty": [10, 4, 10],
        "bid_px": [[100, 99], [100, 99], [99, 98]],
        "ask_px": [[101, 102], [101, 102], [99.8, 100]],
    })
    sig_exit = strat.on_bar({"window": df_exit})
    assert sig_exit is not None and sig_exit.side == "sell" and sig_exit.reduce_only


def test_time_exit():
    df_entry = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 4],
        "bid_px": [[100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2, tp_pct=0.05, sl_pct=0.05, max_hold=1, dynamic_thresholds=False)
    sig = strat.on_bar({"window": df_entry})
    assert sig is not None and sig.side == "buy"

    df_exit = pd.DataFrame({
        "bid_qty": [10, 10, 10],
        "ask_qty": [10, 4, 10],
        "bid_px": [[100, 99], [100, 99], [100, 99]],
        "ask_px": [[101, 102], [101, 102], [101, 102]],
    })
    sig_exit = strat.on_bar({"window": df_exit})
    assert sig_exit is not None and sig_exit.side == "sell" and sig_exit.reduce_only


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
