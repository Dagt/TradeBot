import pytest
import tradingbot.apps.api.main as api_main


@pytest.mark.asyncio
async def test_update_bot_stats_events():
    api_main._BOTS.clear()
    api_main._BOTS[1] = {"stats": {}}
    await api_main.update_bot_stats(1, {"event": "order"})
    await api_main.update_bot_stats(
        1,
        {
            "event": "fill",
            "qty": 1,
            "fee": 0.2,
            "slippage_bps": 5,
            "maker": True,
            "price": 100,
            "pnl": 2,
        },
    )
    await api_main.update_bot_stats(
        1,
        {
            "event": "trade",
            "pnl": 5,
            "trade_pnl": 5,
            "trades_closed": 1,
            "trades_won": 1,
            "hit_rate": 100.0,
        },
    )
    await api_main.update_bot_stats(
        1,
        {
            "event": "trade",
            "pnl": 7,
            "trade_pnl": 7,
            "trades_closed": 2,
            "trades_won": 2,
            "hit_rate": 100.0,
        },
    )
    await api_main.update_bot_stats(1, {"event": "cancel"})
    stats = api_main._BOTS[1]["stats"]
    assert stats["orders"] == 1
    assert stats["fills"] == 1
    assert stats["fees_usd"] == 0.2
    assert stats["slippage_bps"] == 5
    assert stats["pnl"] == pytest.approx(14)
    assert stats["cancels"] == 1
    assert stats["cancel_ratio"] == 1.0
    assert stats["trades_closed"] == 2
    assert stats["trades_won"] == 2
    assert stats["hit_rate"] == pytest.approx(100.0)
    assert "trades_processed" not in stats


@pytest.mark.asyncio
async def test_trade_pnl_accumulates_consecutive_wins():
    api_main._BOTS.clear()
    api_main._BOTS[2] = {"stats": {}}
    first_trade = {
        "event": "trade",
        "pnl": 5,
        "trade_pnl": 5,
        "trades_closed": 1,
        "trades_won": 1,
        "hit_rate": 100.0,
    }
    second_trade = {
        "event": "trade",
        "pnl": 7,
        "trade_pnl": 7,
        "trades_closed": 2,
        "trades_won": 2,
        "hit_rate": 100.0,
    }

    await api_main.update_bot_stats(2, first_trade)
    stats = api_main._BOTS[2]["stats"]
    assert stats["pnl"] == pytest.approx(5)
    assert stats["trades_won"] == 1
    assert stats["hit_rate"] == pytest.approx(100.0)

    await api_main.update_bot_stats(2, second_trade)
    stats = api_main._BOTS[2]["stats"]
    assert stats["pnl"] == pytest.approx(12)
    assert stats["trades_won"] == 2
    assert stats["hit_rate"] == pytest.approx(100.0)

    await api_main.update_bot_stats(2, {"pnl": 12})
    stats = api_main._BOTS[2]["stats"]
    assert stats["pnl"] == pytest.approx(12)


@pytest.mark.asyncio
async def test_partial_fills_and_cancel_tracking():
    api_main._BOTS.clear()
    api_main._BOTS[3] = {"stats": {}}

    await api_main.update_bot_stats(3, {"event": "order"})

    partial_fill = {
        "event": "fill",
        "order_id": "order-fill",
        "pending_qty": 0.5,
        "filled_qty": 0.5,
        "qty": 0.5,
        "fee": 0.1,
        "slippage_bps": 1.0,
        "maker": True,
    }
    await api_main.update_bot_stats(3, partial_fill)

    final_fill = {
        "event": "fill",
        "order_id": "order-fill",
        "pending_qty": 0.0,
        "filled_qty": 0.5,
        "qty": 0.5,
        "fee": 0.1,
        "slippage_bps": 1.0,
        "maker": False,
    }
    await api_main.update_bot_stats(3, final_fill)

    stats = api_main._BOTS[3]["stats"]
    assert stats["fills"] == 1
    assert stats["fees_usd"] == pytest.approx(0.2)

    await api_main.update_bot_stats(3, {"event": "order"})
    await api_main.update_bot_stats(
        3,
        {
            "event": "cancel",
            "order_id": "order-cancel",
            "pending_qty": 1.0,
            "filled_qty": 0.0,
        },
    )

    stats = api_main._BOTS[3]["stats"]
    assert stats["cancels"] == 1
    assert stats["orders"] == 2
    assert stats["cancel_ratio"] == pytest.approx(0.5)

    await api_main.update_bot_stats(
        3,
        {
            "event": "cancel",
            "order_id": "order-fill",
            "pending_qty": 0.0,
            "filled_qty": 0.25,
        },
    )

    stats = api_main._BOTS[3]["stats"]
    assert stats["cancels"] == 1
    assert api_main._BOTS[3].get("_pending_by_order") == {}
