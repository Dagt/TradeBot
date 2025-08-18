import pandas as pd
import pytest


@pytest.fixture
def synthetic_trades():
    """Synthetic trade stream for testing."""
    return pd.DataFrame({
        "ts": [1, 2, 3, 4],
        "price": [100.0, 100.5, 101.0, 99.5],
        "qty": [0.1, 0.2, 0.3, 0.4],
        "side": ["buy", "sell", "buy", "sell"],
    })


@pytest.fixture
def synthetic_l2():
    """Simple level-2 order book snapshot."""
    bids = pd.DataFrame({
        "price": [99.5, 99.0, 98.5],
        "qty": [1.0, 0.8, 0.5],
    })
    asks = pd.DataFrame({
        "price": [100.5, 101.0, 101.5],
        "qty": [1.1, 0.9, 0.4],
    })
    return {"bids": bids, "asks": asks}


@pytest.fixture
def synthetic_volatility():
    """Synthetic volatility value for tests."""
    return 0.04


class DummyMarketAdapter:
    """Simple adapter emulating spot/perp streams and orders for tests."""

    def __init__(self, name: str, trades: list[dict]):
        self.name = name
        self._trades = trades
        self.orders: list[dict] = []
        self.state = type("S", (), {"last_px": {}})

    async def stream_trades(self, symbol: str):
        for t in self._trades:
            self.state.last_px[symbol] = t["price"]
            yield t

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
    ) -> dict:
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        return {"status": "filled", "price": self.state.last_px[symbol]}


@pytest.fixture
def dual_testnet():
    """Return paired spot/perp adapters with simple trade streams."""

    spot_trades = [{"ts": 0, "price": 100.0, "qty": 1.0, "side": "buy"}]
    perp_trades = [{"ts": 0, "price": 101.0, "qty": 1.0, "side": "buy"}]
    spot = DummyMarketAdapter("spot", spot_trades)
    perp = DummyMarketAdapter("perp", perp_trades)
    return spot, perp
