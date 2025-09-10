import asyncio
import signal
from fastapi.testclient import TestClient

from tradingbot.apps.api.main import app


def test_bot_endpoints(monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)

    class DummyProc:
        def __init__(self, pid: int):
            self.pid = pid
            self.returncode = None
            self.stdout = None
            self.stderr = None
            loop = asyncio.get_event_loop()
            self._waiter = loop.create_future()

        async def wait(self):
            await self._waiter
            return self.returncode

        def terminate(self):
            self.returncode = 0
            if not self._waiter.done():
                self._waiter.set_result(None)

        def send_signal(self, sig):
            if sig == signal.SIGSTOP:
                self.returncode = -sig
                if not self._waiter.done():
                    self._waiter.set_result(None)
            elif sig == signal.SIGCONT:
                self.returncode = None
                if self._waiter.done():
                    loop = asyncio.get_event_loop()
                    self._waiter = loop.create_future()
            return None

    procs: list[DummyProc] = []
    calls = {}

    async def fake_exec(*args, **kwargs):
        calls["args"] = args
        proc = DummyProc(100 + len(procs))
        procs.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    payload = {
        "strategy": "dummy",
        "pairs": ["BTC/USDT"],
        "venue": "binance_spot",
        "leverage": 1,
        "risk_pct": 0.03,
        "testnet": True,
        "dry_run": False,
    }

    resp = client.post("/bots", json=payload, auth=("admin", "admin"))
    assert resp.status_code == 200
    pid = resp.json()["pid"]
    argv = list(calls["args"])
    assert "--venue" in argv and "binance_spot" in argv
    assert "--risk-pct" in argv and "0.03" in argv

    lst = client.get("/bots", auth=("admin", "admin"))
    assert lst.status_code == 200
    bots = lst.json()["bots"]
    assert any(b["pid"] == pid for b in bots)
    assert any(b["pid"] == pid and b.get("risk_pct") == 0.03 for b in bots)

    assert client.post(f"/bots/{pid}/pause", auth=("admin", "admin")).status_code == 200
    lst = client.get("/bots", auth=("admin", "admin"))
    assert any(b["pid"] == pid and b["status"] == "paused" for b in lst.json()["bots"])
    assert client.post(f"/bots/{pid}/resume", auth=("admin", "admin")).status_code == 200
    lst = client.get("/bots", auth=("admin", "admin"))
    assert any(b["pid"] == pid and b["status"] == "running" for b in lst.json()["bots"])


def test_cross_arbitrage_start(monkeypatch):
    client = TestClient(app)

    calls = {}

    class DummyProc:
        def __init__(self):
            self.pid = 321
            self.returncode = None

        async def wait(self):
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

    async def fake_exec(*args, **kwargs):
        calls["args"] = args
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    payload = {
        "strategy": "cross_arbitrage",
        "pairs": ["BTC/USDT"],
        "spot": "binance_spot",
        "perp": "binance_futures",
        "threshold": 0.001,
    }

    resp = client.post("/bots", json=payload, auth=("admin", "admin"))
    assert resp.status_code == 200
    assert resp.json()["pid"] == 321
    argv = list(calls["args"])
    assert "run-cross-arb" in argv
    assert "binance_spot" in argv and "binance_futures" in argv
    assert "--notional" not in argv


def test_dashboard_bot_controls():
    client = TestClient(app)
    resp = client.get("/bots", headers={"Accept": "text/html"}, auth=("admin", "admin"))
    assert resp.status_code == 200
    html = resp.text
    assert "haltBot" in html
    assert "killBot" in html
    assert "flattenBot" not in html
    assert "reloadBot" not in html
    for path in ["halt", "flatten", "reload", "kill"]:
        r = client.post(f"/bots/123/{path}", auth=("admin", "admin"))
        assert r.status_code == 404
