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
        "trade_qty": 1.0,
        "leverage": 1,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "risk_pct": 0.03,
        "max_drawdown_pct": 0.1,
        "testnet": True,
        "dry_run": False,
    }

    resp = client.post("/bots", json=payload, auth=("admin", "admin"))
    assert resp.status_code == 200
    pid = resp.json()["pid"]
    argv = list(calls["args"])
    assert "--venue" in argv and "binance_spot" in argv
    assert "--stop-loss" in argv and "0.02" in argv
    assert "--take-profit" in argv and "0.05" in argv
    assert "--risk-pct" in argv and "0.03" in argv
    assert "--max-drawdown-pct" in argv and "0.1" in argv

    lst = client.get("/bots", auth=("admin", "admin"))
    assert lst.status_code == 200
    assert any(b["pid"] == pid for b in lst.json()["bots"])

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
        "notional": 25.0,
        "threshold": 0.001,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "risk_pct": 0.03,
        "max_drawdown_pct": 0.1,
    }

    resp = client.post("/bots", json=payload, auth=("admin", "admin"))
    assert resp.status_code == 200
    assert resp.json()["pid"] == 321
    argv = list(calls["args"])
    assert "run-cross-arb" in argv
    assert "binance_spot" in argv and "binance_futures" in argv

