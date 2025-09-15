import pytest
import tradingbot.apps.api.main as api_main


@pytest.mark.asyncio
async def test_initial_metrics_zero():
    api_main._BOTS.clear()
    try:
        api_main._BOTS[1] = {"stats": {}}
        await api_main.update_bot_stats(1)
        stats = api_main._BOTS[1]["stats"]
        assert stats["hit_rate"] == 0.0
        assert stats["cancel_ratio"] == 0.0
    finally:
        api_main._BOTS.clear()

