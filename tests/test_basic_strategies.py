import pandas as pd
import pytest

from tradingbot.strategies import momentum, mean_reversion
import tradingbot.strategies.arbitrage as arbitrage


def backtest(data: pd.DataFrame, signals: pd.DataFrame, price_col: str) -> float:
    returns = data[price_col].pct_change().fillna(0)
    pnl = signals["position"] * returns - signals["fee"] - signals["slippage"]
    return pnl.cumsum().iloc[-1]


def test_momentum_backtest():
    data = pd.DataFrame({"price": list(range(1, 11))})
    base_params = {
        "window": 2,
        "position_size": 1,
        "stop_loss": 0.05,
        "take_profit": 0.1,
    }
    signals = momentum.generate_signals(data, {**base_params, "fee": 0.0, "slippage": 0.0})
    pnl_no_fee = backtest(data, signals, "price")
    assert signals["stop_loss"].iloc[0] == pytest.approx(data["price"].iloc[0] * (1 - base_params["stop_loss"]))
    pnl_fee = backtest(
        data,
        momentum.generate_signals(data, {**base_params, "fee": 0.01, "slippage": 0.01}),
        "price",
    )
    assert pnl_fee <= pnl_no_fee


def test_mean_reversion_backtest():
    data = pd.DataFrame({"price": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]})
    base_params = {
        "window": 2,
        "threshold": 0.1,
        "position_size": 1,
        "stop_loss": 0.05,
        "take_profit": 0.1,
    }
    signals = mean_reversion.generate_signals(data, {**base_params, "fee": 0.0, "slippage": 0.0})
    pnl_no_fee = backtest(data, signals, "price")
    assert signals["stop_loss"].iloc[0] == pytest.approx(data["price"].iloc[0] * (1 - base_params["stop_loss"]))
    pnl_fee = backtest(
        data,
        mean_reversion.generate_signals(data, {**base_params, "fee": 0.01, "slippage": 0.01}),
        "price",
    )
    assert pnl_fee <= pnl_no_fee


def test_arbitrage_backtest():
    data = pd.DataFrame(
        {"asset_a": [1, 2, 1, 2, 1, 2, 1, 2], "asset_b": [1, 1, 1, 1, 1, 1, 1, 1]}
    )
    base_params = {
        "threshold": 0.0,
        "position_size": 1,
        "stop_loss": 0.05,
        "take_profit": 0.1,
    }
    signals = arbitrage.generate_signals(data, {**base_params, "fee": 0.0, "slippage": 0.0})
    pnl_no_fee = backtest(data, signals, "asset_a")
    assert signals["stop_loss"].iloc[0] == pytest.approx(data["asset_a"].iloc[0] * (1 - base_params["stop_loss"]))
    pnl_fee = backtest(
        data,
        arbitrage.generate_signals(data, {**base_params, "fee": 0.01, "slippage": 0.01}),
        "asset_a",
    )
    assert pnl_fee <= pnl_no_fee

