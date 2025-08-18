import asyncio
import pytest

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.adapters.bybit_futures import BybitFuturesAdapter, ccxt as bybit_ccxt
from tradingbot.adapters.okx_spot import OKXSpotAdapter

NetworkError = getattr(bybit_ccxt, "NetworkError", Exception)


class FlakyRest:
    """REST stub failing once with ``NetworkError`` then succeeding."""

    def __init__(self):
        self.order_attempts = 0
        self.cancel_attempts = 0

    def create_order(self, symbol, type_, side, qty, price=None, params=None):
        self.order_attempts += 1
        if self.order_attempts == 1:
            raise NetworkError("boom")
        return {"id": "1", "status": "open", "symbol": symbol, "qty": qty, "price": price}

    def cancel_order(self, order_id, symbol=None, params=None):
        self.cancel_attempts += 1
        if self.cancel_attempts == 1:
            raise NetworkError("boom")
        return {"id": order_id, "status": "canceled"}


@pytest.mark.asyncio
async def test_bybit_place_cancel_retry(monkeypatch, synthetic_l2):
    async def fast_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    adapter = BybitFuturesAdapter.__new__(BybitFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = FlakyRest()

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    price = float(synthetic_l2["asks"]["price"].iloc[0])
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1, price=price)
    assert res["status"] == "open"
    cancel = await adapter.cancel_order(res["id"], "BTC/USDT")
    assert cancel["status"] == "canceled"
    assert adapter.rest.order_attempts == 2
    assert adapter.rest.cancel_attempts == 2


@pytest.mark.asyncio
async def test_okx_place_cancel_retry(monkeypatch, synthetic_trades):
    async def fast_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    adapter = OKXSpotAdapter.__new__(OKXSpotAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = FlakyRest()

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    qty = float(synthetic_trades.qty.sum())
    res = await adapter.place_order("BTC/USDT", "buy", "market", qty)
    assert res["symbol"] == "BTC/USDT"
    cancel = await adapter.cancel_order(res["id"], "BTC/USDT")
    assert cancel["status"] == "canceled"
    assert adapter.rest.order_attempts == 2
    assert adapter.rest.cancel_attempts == 2
