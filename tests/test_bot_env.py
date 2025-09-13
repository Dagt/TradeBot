import asyncio
from pathlib import Path

import pytest

from tradingbot.apps.api import main as api_main


class DummyProc:
    def __init__(self, done: bool = False):
        self.pid = 999
        self.returncode = None
        self.stdout = None
        self.stderr = None
        self._done = asyncio.Event()
        if done:
            self.returncode = 0
            self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode


@pytest.mark.asyncio
async def test_start_bot_inherits_env_and_running(monkeypatch):
    monkeypatch.setenv("PGUSER", "alice")
    monkeypatch.setattr(api_main, "BOT_LOG_RETENTION", 0)

    captured: dict[str, object] = {}

    async def fake_exec(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        proc = DummyProc()
        captured["proc"] = proc
        return proc

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

    status = await api_main.list_bots(DummyReq())
    bot = status["bots"][0]
    assert bot["status"] == "running"
    assert bot["stats"] == {}

    # cleanup
    captured["proc"]._done.set()  # type: ignore[index]
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_update_bot_stats(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(api_main, "BOT_LOG_RETENTION", 0)

    async def fake_exec(*args, **kwargs):
        proc = DummyProc()
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(api_main, "_build_bot_args", lambda cfg, params=None: ["echo", "hi"])

    api_main._BOTS.clear()

    cfg = api_main.BotConfig(strategy="dummy")
    await api_main.start_bot(cfg)
    await api_main.update_bot_stats(999, orders=5, fills=2, exposure=1.5)

    class DummyReq:
        headers = {}

    data = await api_main.list_bots(DummyReq())
    stats = data["bots"][0]["stats"]
    assert stats["orders"] == 5
    assert stats["fills"] == 2
    assert stats["exposure"] == 1.5

    # cleanup
    captured["proc"]._done.set()  # type: ignore[index]
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_finished_bot_removed(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_exec(*args, **kwargs):
        proc = DummyProc(done=True)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(api_main, "_build_bot_args", lambda cfg, params=None: ["echo", "hi"])
    # remove quickly for test
    monkeypatch.setattr(api_main, "BOT_LOG_RETENTION", 0)

    api_main._BOTS.clear()

    cfg = api_main.BotConfig(strategy="dummy")
    await api_main.start_bot(cfg)

    await asyncio.sleep(0.05)

    class DummyReq:
        headers = {}

    data = await api_main.list_bots(DummyReq())
    assert data["bots"] == []
