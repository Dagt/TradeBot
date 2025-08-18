import datetime as dt
import pytest

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.core import normalize


class DummyAdapter(ExchangeAdapter):
    async def stream_trades(self, symbol: str):
        if False:
            yield {}
        return

    async def place_order(self, *args, **kwargs):
        return {}

    async def cancel_order(self, order_id: str):
        return {}


@pytest.mark.asyncio
async def test_normalize_helpers():
    ad = DummyAdapter()
    assert normalize("BTC/USDT") == "BTCUSDT"
    ts = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    book = ad.normalize_order_book("BTCUSDT", ts, [[1.0, 2.0]], [[3.0, 4.0]])
    assert book["bid_px"] == [1.0]
    assert book["ask_qty"] == [4.0]
