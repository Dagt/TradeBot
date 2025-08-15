import pytest


@pytest.mark.asyncio
async def test_paper_adapter_execution(paper_adapter, mock_order):
    paper_adapter.update_last_price(mock_order["symbol"], 100.0)

    res = await paper_adapter.place_order(**mock_order)
    assert res["status"] == "filled"
    assert paper_adapter.state.pos[mock_order["symbol"]].qty == mock_order["qty"]

    cancel = await paper_adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"
