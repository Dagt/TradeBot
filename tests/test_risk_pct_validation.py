import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine


@pytest.fixture
def dummy_data():
    return {"BTC/USDT": pd.DataFrame({
        "open": [1.0],
        "high": [1.0],
        "low": [1.0],
        "close": [1.0],
        "volume": [1.0],
    })}


def test_engine_normalizes_percentage(dummy_data):
    eng = EventDrivenBacktestEngine(dummy_data, [("breakout_atr", "BTC/USDT")], risk_pct=5)
    assert eng._risk_pct == pytest.approx(0.05)
    eng = EventDrivenBacktestEngine(dummy_data, [("breakout_atr", "BTC/USDT")], risk_pct=1)
    assert eng._risk_pct == pytest.approx(0.01)


def test_engine_rejects_invalid_risk_pct(dummy_data):
    with pytest.raises(ValueError):
        EventDrivenBacktestEngine(dummy_data, [("breakout_atr", "BTC/USDT")], risk_pct=-0.1)
    with pytest.raises(ValueError):
        EventDrivenBacktestEngine(dummy_data, [("breakout_atr", "BTC/USDT")], risk_pct=150)
