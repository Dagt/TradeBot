import pandas as pd

from tradingbot.strategies.depth_imbalance import DepthImbalance


def test_depth_imbalance_strategy_buy():
    df = pd.DataFrame({"bid_qty": [5, 7, 9], "ask_qty": [5, 5, 3]})
    strat = DepthImbalance(window=2, threshold=0.1)
    sig = strat.on_bar({"window": df})
    assert sig is not None and sig.side == "buy"


def test_depth_imbalance_strategy_sell():
    df = pd.DataFrame({"bid_qty": [5, 4, 3], "ask_qty": [5, 6, 7]})
    strat = DepthImbalance(window=2, threshold=0.1)
    sig = strat.on_bar({"window": df})
    assert sig is not None and sig.side == "sell"
