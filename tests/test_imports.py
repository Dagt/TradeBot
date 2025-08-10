"""Basic import tests for TradeBot modules."""
import pandas as pd
import tradebot
from tradebot import (backtesting, cli, config, data, execution, features,
                      monitoring, persistence, risk, strategies)


def test_imports() -> None:
    assert tradebot is not None
    assert cli is not None
    assert config.Settings().mode == "paper"
    assert features.compute_rsi(pd.Series([1, 2, 3]), period=2) is not None
    assert strategies.Strategy("demo").name == "demo"
    assert risk.RiskManager().check_position(0.5)
    assert execution.OrderRouter() is not None
    assert persistence.Database().dsn.startswith("postgresql")
    assert backtesting.Backtester([1, 2, 3]).run() == 6
    monitor = monitoring.Monitor()
    monitor.emit("test", 1.0)
