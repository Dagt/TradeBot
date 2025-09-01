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


@pytest.mark.parametrize(
    "risk_pct,expected",
    [
        (0.5, 0.5),  # already a fraction
        (1, 1.0),  # 100%
        (5, 0.05),  # percentage expressed as integer
        (50, 0.5),  # percentage conversion boundary
    ],
)
def test_engine_normalizes_percentage(dummy_data, risk_pct, expected):
    eng = EventDrivenBacktestEngine(dummy_data, [("breakout_atr", "BTC/USDT")], risk_pct=risk_pct)
    assert eng._risk_pct == pytest.approx(expected)


@pytest.mark.parametrize("risk_pct", [-0.1, 120])
def test_engine_rejects_invalid_risk_pct(dummy_data, risk_pct):
    with pytest.raises(ValueError):
        EventDrivenBacktestEngine(dummy_data, [("breakout_atr", "BTC/USDT")], risk_pct=risk_pct)
