import asyncio
import os
from pathlib import Path

import pytest

from tradingbot.apps.api import main as api_main


class DummyProc:
    def __init__(self):
        self.pid = 999
        self.returncode = None
        self.stdout = None
        self.stderr = None

    async def wait(self):
        return None


@pytest.mark.asyncio
async def test_start_bot_inherits_env_and_running(monkeypatch):
    monkeypatch.setenv("PGUSER", "alice")

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(api_main, "_build_bot_args", lambda cfg, params=None: ["echo", "hi"])

    api_main._BOTS.clear()

    cfg = api_main.BotConfig(strategy="dummy")
    await api_main.start_bot(cfg)

    env = captured["env"]
    assert env["PGUSER"] == "alice"
    repo_root = Path(api_main.__file__).resolve().parents[3]
    assert env["PYTHONPATH"].startswith(str(repo_root))

    class DummyReq:
        headers = {}

    status = api_main.list_bots(DummyReq())
    bot = status["bots"][0]
    assert bot["status"] == "running"
    assert bot["stats"] == {}


@pytest.mark.asyncio
async def test_update_bot_stats(monkeypatch):
    async def fake_exec(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(api_main, "_build_bot_args", lambda cfg, params=None: ["echo", "hi"])

    api_main._BOTS.clear()

    cfg = api_main.BotConfig(strategy="dummy")
    await api_main.start_bot(cfg)
    api_main.update_bot_stats(999, orders_sent=5, fills=2, inventory=1.5)

    class DummyReq:
        headers = {}

    data = api_main.list_bots(DummyReq())
    stats = data["bots"][0]["stats"]
    assert stats["orders_sent"] == 5
    assert stats["fills"] == 2
    assert stats["inventory"] == 1.5
