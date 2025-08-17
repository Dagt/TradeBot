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
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=2)
    sig = strat.on_bar({"window": df})
    assert sig is not None and sig.side == "buy"


def test_liquidity_events_strategy_sell_gap():
    df = pd.DataFrame({
        "bid_qty": [10, 10],
        "ask_qty": [10, 10],
        "bid_px": [[100, 99], [100, 90]],
        "ask_px": [[101, 102], [101, 102]],
    })
    strat = LiquidityEvents(vacuum_threshold=0.5, gap_threshold=5)
    sig = strat.on_bar({"window": df})
    assert sig is not None and sig.side == "sell"
