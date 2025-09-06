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
    assert status["bots"][0]["status"] == "running"
