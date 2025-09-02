import pandas as pd
from tradingbot.strategies.order_flow import OrderFlow


def test_order_flow_signal_1m():
    df = pd.DataFrame({
        "bid_qty": [1, 2, 3, 4, 10],
        "ask_qty": [5, 4, 3, 2, 1],
        "close": [100, 101, 100, 102, 101],
    })
    strat = OrderFlow(window=3, buy_threshold=10.0, sell_threshold=10.0)
    sig = strat.on_bar({"window": df, "timeframe": "1m", "symbol": "X", "volatility": 0.0})
    assert sig is not None and sig.side == "buy"


def test_order_flow_signal_15m():
    df = pd.DataFrame({
        "bid_qty": [10, 4, 3, 2, 1],
        "ask_qty": [1, 2, 3, 4, 10],
        "close": [101, 100, 102, 99, 98],
    })
    strat = OrderFlow(window=3, buy_threshold=10.0, sell_threshold=10.0)
    sig = strat.on_bar({"window": df, "timeframe": "15m", "symbol": "X", "volatility": 0.0})
    assert sig is not None and sig.side == "sell"

