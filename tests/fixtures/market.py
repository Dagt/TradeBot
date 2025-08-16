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
