import tradingbot.apps.api.main as api_main


def test_update_bot_stats_events():
    api_main._BOTS.clear()
    api_main._BOTS[1] = {"stats": {}}
    api_main.update_bot_stats(1, {"event": "order"})
    api_main.update_bot_stats(
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
    api_main.update_bot_stats(1, {"event": "trade", "pnl": 5})
    api_main.update_bot_stats(1, {"event": "trade", "pnl": 7})
    stats = api_main._BOTS[1]["stats"]
    assert stats["orders_sent"] == 1
    assert stats["fills"] == 1
    assert stats["fees_usd"] == 0.2
    assert stats["hit_rate"] == 1.0
    assert stats["slippage_bps"] == 5
    assert stats["pnl"] == 9
    assert "trades_processed" not in stats
