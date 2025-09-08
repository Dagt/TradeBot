import asyncio
from fastapi.testclient import TestClient

from tradingbot.apps.api.main import app


def test_bot_endpoints(monkeypatch):
    client = TestClient(app)

    class DummyProc:
        def __init__(self, pid: int):
            self.pid = pid
            self.returncode = None

        async def wait(self):
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

        def send_signal(self, sig):
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
    data = lst.json()["bots"]
    assert any(b["pid"] == pid for b in data)
    bot = next(b for b in data if b["pid"] == pid)
    assert bot.get("risk_pct") == payload["risk_pct"]

    assert client.post(f"/bots/{pid}/pause", auth=("admin", "admin")).status_code == 200
    assert client.post(f"/bots/{pid}/resume", auth=("admin", "admin")).status_code == 200
    assert client.post(f"/bots/{pid}/stop", auth=("admin", "admin")).status_code == 200
    assert client.delete(f"/bots/{pid}", auth=("admin", "admin")).status_code == 200


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
