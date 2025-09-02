import pandas as pd
from tradingbot.strategies import mean_reversion as mr
from tradingbot.strategies.mean_reversion import MeanReversion, generate_signals

def test_mean_reversion_on_bar_signals():
    df_down = pd.DataFrame({"close": list(range(20, 0, -1))})
    df_up = pd.DataFrame({"close": list(range(1, 21))})
    strat = MeanReversion(rsi_n=5)
    sig_buy = strat.on_bar({"window": df_down, "volatility": 0.0})
    sig_sell = strat.on_bar({"window": df_up, "volatility": 0.0})
    assert sig_buy.side == "buy"
    assert sig_sell.side == "sell"

def test_mean_reversion_generate_signals():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 3, 2, 1, 2, 3]})
    params = {"window": 3, "threshold": 0.5, "position_size": 1}
    res = generate_signals(df, params)
    assert {"signal", "position", "fee", "slippage"} <= set(res.columns)
    assert len(res) == len(df)


def _const_rsi(val: float):
    def fn(df: pd.DataFrame, n: int):  # noqa: ANN001
        return pd.Series([val] * len(df), index=df.index)
    return fn


def test_trend_detection_1m(monkeypatch):
    df = pd.DataFrame({"close": list(range(1, 100))})
    monkeypatch.setattr(mr, "rsi", _const_rsi(56))
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (55, 45))
    strat = MeanReversion(timeframe="1m")
    sig = strat.on_bar({"window": df})
    assert sig is None


def test_trend_detection_5m(monkeypatch):
    df = pd.DataFrame({"close": list(range(1, 100))})
    monkeypatch.setattr(mr, "rsi", _const_rsi(56))
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (55, 45))
    strat = MeanReversion(timeframe="5m")
    sig = strat.on_bar({"window": df})
    assert sig is None
