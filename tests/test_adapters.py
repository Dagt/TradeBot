import pytest


async def collect_trades(adapter):
    trades = []
    async for t in adapter.stream_trades("BTCUSDT"):
        trades.append(t)
    return trades


@pytest.mark.asyncio
async def test_mock_adapter_stream_and_orders(mock_adapter, mock_trades, mock_order):
    collected = await collect_trades(mock_adapter)
    assert collected == mock_trades

    order_res = await mock_adapter.place_order(**mock_order)
    assert order_res["status"] == "placed"
    assert order_res["symbol"] == mock_order["symbol"]

    cancel = await mock_adapter.cancel_order("1")
    assert cancel["status"] == "canceled"
