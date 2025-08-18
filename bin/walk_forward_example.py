#!/usr/bin/env python3
"""Example script running walk-forward analysis with vectorbt."""

import numpy as np
import pandas as pd
import vectorbt as vbt

from tradingbot.backtest.vectorbt_engine import walk_forward


class MAStrategy:
    @staticmethod
    def signal(close, fast, slow):
        fast_ma = vbt.MA.run(close, fast)
        slow_ma = vbt.MA.run(close, slow)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits


def main() -> None:
    data = pd.read_csv("data/examples/btcusdt_1m.csv")
    params = {"fast": [5, 10], "slow": [20]}
    res = walk_forward(data, MAStrategy, params, train_size=60, test_size=30)
    print(res)


if __name__ == "__main__":
    main()
