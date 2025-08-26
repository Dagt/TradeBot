import sys
import pathlib
import pytest

sys.path.append(str(pathlib.Path(__file__).parent))
root_dir = pathlib.Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
# Ensure the src package is importable
sys.path.append(str(root_dir / "src"))

from fixtures.market import (  # noqa: E402, F401
    synthetic_trades,
    synthetic_l2,
    synthetic_volatility,
    dual_testnet,
)

class ExchangeAdapter:  # type: ignore
    """Stub used for tests when exchange adapters are unavailable."""
    pass
from tradingbot.execution.paper import PaperAdapter  # noqa: E402


class DummyAdapter(ExchangeAdapter):
    def __init__(self, trades):
        self._trades = trades

    async def stream_trades(self, symbol: str):
        for trade in self._trades:
            yield trade

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
        iceberg_qty: float | None = None,
    ):
        return {
            "status": "placed",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
        }

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

    equity = 10_000.0
    position_pct = 0.10  # 10% del equity
    risk_pct = 0.02      # arriesgar 2% del equity invertido
    price = 100.0
    stop_loss_pct = risk_pct / position_pct

    rm = RiskManager(
        equity_pct=position_pct,
        equity_actual=equity,
        stop_loss_pct=stop_loss_pct,
    )
    # Atributos auxiliares para las pruebas
    rm.price = price
    rm.position_pct = position_pct
    rm.risk_pct = risk_pct
    return rm


@pytest.fixture
def portfolio_guard():
    from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig

    equity = 10_000.0
    total_pct = 0.50       # 50% del equity total
    per_symbol_pct = 0.20  # 20% por símbolo

    cfg = GuardConfig(
        total_cap_usdt=equity * total_pct,
        per_symbol_cap_usdt=equity * per_symbol_pct,
        venue="test",
    )

    guard = PortfolioGuard(cfg)
    guard.equity = equity
    guard.total_pct = total_pct
    guard.per_symbol_pct = per_symbol_pct
    return guard
