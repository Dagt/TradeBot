import sys
import pathlib
import pytest

# Ensure the src package is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.execution.paper import PaperAdapter


class DummyAdapter(ExchangeAdapter):
    def __init__(self, trades):
        self._trades = trades

    async def stream_trades(self, symbol: str):
        for trade in self._trades:
            yield trade

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, price: float | None = None):
        return {"status": "placed", "symbol": symbol, "side": side, "qty": qty, "price": price}

    async def cancel_order(self, order_id: str):
        return {"status": "canceled", "order_id": order_id}


@pytest.fixture
def mock_trades():
    return [
        {"ts": 1, "price": 100.0, "qty": 1.0, "side": "buy"},
        {"ts": 2, "price": 101.0, "qty": 2.0, "side": "sell"},
    ]


@pytest.fixture
def mock_order():
    return {"symbol": "BTCUSDT", "side": "buy", "type_": "market", "qty": 1.0}


@pytest.fixture
def mock_adapter(mock_trades):
    return DummyAdapter(mock_trades)


@pytest.fixture
def paper_adapter():
    return PaperAdapter()


@pytest.fixture
def breakout_df_buy():
    import pandas as pd
    data = {
        "open": [1, 2, 3, 4],
        "high": [2, 3, 4, 5],
        "low": [0.5, 1.5, 2.5, 3.5],
        "close": [1.5, 2.5, 3.5, 8.0],
        "volume": [1, 1, 1, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def breakout_df_sell():
    import pandas as pd
    data = {
        "open": [1, 2, 3, 4],
        "high": [2, 3, 4, 5],
        "low": [0.5, 1.5, 2.5, 3.5],
        "close": [1.5, 2.5, 3.5, -2.0],
        "volume": [1, 1, 1, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def risk_manager():
    from tradingbot.risk.manager import RiskManager

    return RiskManager(max_pos=5)
