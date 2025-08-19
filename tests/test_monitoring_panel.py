import asyncio
from fastapi.testclient import TestClient

from monitoring.panel import app


def test_config_roundtrip():
    client = TestClient(app)
    resp = client.get("/config")
    assert resp.status_code == 200
    assert "config" in resp.json()

    payload = {"strategy": "mean_reversion", "pairs": ["BTC/USDT"], "notional": 50}
    post_resp = client.post("/config", json=payload)
    assert post_resp.status_code == 200
    post_cfg = post_resp.json()["config"]
    for key, value in payload.items():
        assert post_cfg[key] == value

    # Roundtrip: ensure the configuration persists
    get_resp = client.get("/config")
    assert get_resp.status_code == 200
    get_cfg = get_resp.json()["config"]
    for key, value in payload.items():
        assert get_cfg[key] == value


def test_start_stop(monkeypatch):
    client = TestClient(app)
    client.post("/config", json={"strategy": "dummy"})

    calls = {}

    class DummyProc:
        def __init__(self):
            self.pid = 123
            self.returncode = None

        async def wait(self):
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

    async def fake_exec(*args, **kwargs):
        calls["args"] = args
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    resp = client.post("/bot/start")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert data["pid"] == 123
    assert calls

    resp = client.post("/bot/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
