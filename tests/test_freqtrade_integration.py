import pandas as pd
import pytest

pytest.importorskip("freqtrade")

from tradingbot.backtesting.freqtrade_wrapper import run_strategy


class DummyStrategy:
    timeframe = "3m"

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe["enter_long"] = dataframe["close"] > dataframe["close"].shift(1)
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_long"] = dataframe["close"] < dataframe["close"].shift(1)
        return dataframe


def test_freqtrade_wrapper_basic():
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.0, 2.0, 1.0, 3.0],
            "volume": [1, 1, 1, 1],
        },
        index=pd.date_range("2021-01-01", periods=4, freq="3T"),
    )
    result = run_strategy(df, DummyStrategy)
    assert "equity" in result
    assert isinstance(result["trades"], list)
