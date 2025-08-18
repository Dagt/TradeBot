import pytest

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.adapters.binance_futures import BinanceFuturesAdapter
from tradingbot.adapters.bybit_futures import BybitFuturesAdapter
from tradingbot.adapters.okx_futures import OKXFuturesAdapter


class DummyRestBinance:
    def __init__(self):
        self.params = None

    async def futures_order_new(self, **params):
        self.params = params
        return {"orderId": "1", "clientOrderId": params.get("newClientOrderId")}


class DummyRestCreate:
    def __init__(self):
        self.args = None

    def create_order(self, symbol, type_, side, qty, price, params):
        self.args = (symbol, type_, side, qty, price, params)
        return {"id": "1"}


@pytest.mark.asyncio
async def test_binance_translates_iceberg():
    adapter = BinanceFuturesAdapter.__new__(BinanceFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = DummyRestBinance()
    adapter.leverage = 1
    adapter.testnet = True
    adapter.taker_fee_bps = 0.0
    res = await adapter.place_order(
        "BTC/USDT", "buy", "limit", 1.0, price=100.0, iceberg_qty=0.5
    )
    assert adapter.rest.params["icebergQty"] == 0.5


@pytest.mark.asyncio
async def test_bybit_translates_iceberg():
    adapter = BybitFuturesAdapter.__new__(BybitFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = DummyRestCreate()
    res = await adapter.place_order(
        "BTC/USDT", "buy", "limit", 1.0, price=100.0, iceberg_qty=0.5
    )
    assert adapter.rest.args[5]["iceberg"] == 0.5


@pytest.mark.asyncio
async def test_okx_translates_iceberg():
    adapter = OKXFuturesAdapter.__new__(OKXFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = DummyRestCreate()
    res = await adapter.place_order(
        "BTC/USDT", "buy", "limit", 1.0, price=100.0, iceberg_qty=0.5
    )
    assert adapter.rest.args[5]["iceberg"] == 0.5
