import pandas as pd
from tradingbot.data.features import order_flow_imbalance, depth_imbalance

def test_order_flow_imbalance():
    df = pd.DataFrame({
        'bid_qty': [5, 7, 3, 4],
        'ask_qty': [8, 6, 5, 7],
    })
    ofi = order_flow_imbalance(df)
    assert list(ofi) == [0, 4, -3, -1]

def test_depth_imbalance():
    df = pd.DataFrame({
        'bid_qty': [5, 7, 3, 4],
        'ask_qty': [8, 6, 5, 7],
    })
    di = depth_imbalance(df)
    expected = [
        (5-8)/(5+8),
        (7-6)/(7+6),
        (3-5)/(3+5),
        (4-7)/(4+7),
    ]
    assert all(abs(a - b) < 1e-9 for a, b in zip(di, expected))
