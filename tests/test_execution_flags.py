import time
import pytest
from tradingbot.execution.venue_adapter import translate_order_flags


class DummyTestnetAdapter:
    """Simple testnet adapter that records latency and order params."""

    def __init__(self, venue: str):
        self.venue = venue
        self.orders: dict[str, dict] = {}

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        **flags,
    ) -> dict:
        start = time.monotonic()
        params = translate_order_flags(self.venue, **flags)
        order_id = f"{symbol}-{len(self.orders) + 1}"
        self.orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "params": params,
        }
        latency = time.monotonic() - start
        return {"order_id": order_id, "status": "open", "params": params, "latency": latency}

    async def cancel_order(self, order_id: str) -> dict:
        self.orders.pop(order_id, None)
        return {"status": "canceled", "order_id": order_id}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "venue,expected",
    [
        ("binance_futures_testnet", {"timeInForce": "GTX"}),
        ("bybit_testnet", {"postOnly": True}),
        ("okx_testnet", {"postOnly": True}),
    ],
)
async def test_post_only_orders(venue, expected):
    adapter = DummyTestnetAdapter(venue)
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1, price=100, post_only=True)
    assert res["status"] == "open"
    assert res["params"] == expected
    assert res["latency"] >= 0
    cancel = await adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"
    assert adapter.orders == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "venue",
    ["binance_futures_testnet", "bybit_testnet", "okx_testnet"],
)
async def test_ioc_orders(venue):
    adapter = DummyTestnetAdapter(venue)
    res = await adapter.place_order(
        "BTC/USDT",
        "buy",
        "limit",
        1,
        price=100,
        time_in_force="IOC",
    )
    assert res["status"] == "open"
    assert res["params"].get("timeInForce") == "IOC"
    cancel = await adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"
    assert adapter.orders == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "venue",
    ["binance_futures_testnet", "bybit_testnet", "okx_testnet"],
)
async def test_fok_orders(venue):
    adapter = DummyTestnetAdapter(venue)
    res = await adapter.place_order(
        "BTC/USDT",
        "buy",
        "limit",
        1,
        price=100,
        time_in_force="FOK",
    )
    assert res["status"] == "open"
    assert res["params"].get("timeInForce") == "FOK"
    cancel = await adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"
    assert adapter.orders == {}


