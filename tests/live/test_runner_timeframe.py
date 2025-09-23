from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.live import runner


class DummyAccount:
    def __init__(self) -> None:
        self.market_type = ""
        self.open_orders: dict[str, dict[str, float]] = {}

    def current_exposure(self, symbol: str) -> tuple[float, float]:
        return 0.0, 0.0

    def update_open_order(self, symbol: str, side: str, qty: float) -> None:
        orders = self.open_orders.setdefault(symbol, {})
        orders[side] = qty


class DummyPaperAdapter:
    def __init__(self, fee_bps: float = 0.0) -> None:
        self.account = DummyAccount()
        self.state = SimpleNamespace(realized_pnl=0.0, last_px={}, order_book={})

    def update_last_price(self, symbol: str, px: float) -> list[dict]:
        self.state.last_px[symbol] = px
        return []

    def equity(self, *_args, **_kwargs) -> float:
        return 1000.0


class DummyExecutionRouter:
    def __init__(self, adapters, prefer: str | None = None) -> None:  # noqa: ANN001
        self.adapters = adapters
        self.prefer = prefer
        self.on_partial_fill = None
        self.on_order_expiry = None

    async def handle_paper_event(self, _event) -> None:  # noqa: ANN001
        return None


class DummyExecBroker:
    def __init__(self, adapter) -> None:  # noqa: ANN001
        self.adapter = adapter
        self.state = SimpleNamespace(realized_pnl=0.0)

    async def place_limit(self, *args, **kwargs):  # noqa: ANN001, D401
        """Minimal async interface returning an empty execution response."""

        return {"filled_qty": 0.0, "pending_qty": 0.0, "realized_pnl": 0.0}


class DummyPortfolioGuard:
    def __init__(self, cfg) -> None:  # noqa: ANN001
        self.cfg = cfg

    def refresh_usd_caps(self, _equity: float) -> None:
        return None


class DummyDailyGuard:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None


class DummyRiskService:
    def __init__(
        self,
        *args,
        account=None,
        **_kwargs,
    ) -> None:  # noqa: ANN001
        self.account = account or DummyAccount()
        self.min_order_qty = 0.0
        self.allow_short = True

    def mark_price(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None

    def daily_mark(self, *args, **kwargs) -> tuple[bool, str]:  # noqa: ANN001
        return False, ""

    def get_trade(self, *args, **kwargs):  # noqa: ANN001
        return None

    def register_order(self, *args, **kwargs) -> bool:  # noqa: ANN001
        return True

    def update_trailing(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None

    def manage_position(self, *args, **kwargs):  # noqa: ANN001
        return None

    def on_fill(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None

    def update_position(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None


class DummyAdapter:
    meta = None

    async def stream_trades(self, symbol):  # noqa: ANN001
        yield {"ts": datetime.now(timezone.utc), "price": 100.0, "qty": 1.0}


class DummyAgg:
    def __init__(self, timeframe: str = "1m") -> None:
        self.timeframe = timeframe
        ts_now = datetime.now(timezone.utc)
        base_bar = SimpleNamespace(ts_open=ts_now, o=100.0, h=100.0, l=100.0, c=100.0, v=1.0)
        self.completed = [base_bar for _ in range(200)]
        self._closed = False

    def on_trade(self, ts, px, qty):  # noqa: ANN001
        if self._closed:
            return None
        self._closed = True
        return SimpleNamespace(ts_open=ts, o=px, h=px, l=px, c=px, v=qty)

    def last_n_bars_df(self, n: int) -> pd.DataFrame:  # noqa: D401
        """Return a constant window to satisfy warm-up checks."""

        rows = list(range(200))
        return pd.DataFrame(
            {
                "timestamp": rows,
                "open": [100.0] * 200,
                "high": [101.0] * 200,
                "low": [99.0] * 200,
                "close": [100.0] * 200,
                "volume": [1.0] * 200,
            }
        )


@pytest.mark.asyncio
async def test_run_live_binance_passes_timeframe(monkeypatch):
    timeframe = "15m"
    captured: list[dict] = []

    class CaptureStrategy:
        pending_qty: dict = {}

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
            self.pending_qty = {}
            self.risk_service = None

        def on_partial_fill(self, *args, **kwargs):  # noqa: ANN001
            return None

        def on_order_expiry(self, *args, **kwargs):  # noqa: ANN001
            return None

        def on_bar(self, bar):  # noqa: ANN001
            captured.append(bar)
            raise RuntimeError("stop")

    monkeypatch.setitem(runner.STRATEGIES, "dummy", CaptureStrategy)
    monkeypatch.setattr(runner, "BinanceWSAdapter", DummyAdapter)
    monkeypatch.setattr(runner, "BarAggregator", DummyAgg)
    monkeypatch.setattr(runner, "PaperAdapter", DummyPaperAdapter)
    monkeypatch.setattr(runner, "ExecutionRouter", DummyExecutionRouter)
    monkeypatch.setattr(runner, "Broker", DummyExecBroker)
    monkeypatch.setattr(runner, "PortfolioGuard", DummyPortfolioGuard)
    monkeypatch.setattr(runner, "DailyGuard", DummyDailyGuard)
    monkeypatch.setattr(runner, "RiskService", DummyRiskService)
    monkeypatch.setattr(runner, "load_positions", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError):
        await runner.run_live_binance(
            symbol="BTC/USDT",
            strategy_name="dummy",
            timeframe=timeframe,
        )

    assert captured, "strategy should receive at least one bar"
    assert captured[0]["timeframe"] == timeframe
    assert captured[0]["symbol"] == "BTC/USDT"
