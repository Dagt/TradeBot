from fastapi.testclient import TestClient

from tradingbot.apps.api.main import app


def _client() -> TestClient:
    """Return a client authenticated with the default credentials."""

    headers = {"Authorization": "Basic YWRtaW46YWRtaW4="}
    return TestClient(app, headers=headers)


def test_smoke():
    assert 2 + 2 == 4


def test_metrics_endpoint_exposed():
    client = _client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "pnl" in data


def test_cli_endpoint_runs_help():
    client = _client()
    resp = client.post("/cli/run", json={"command": "--help"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["returncode"] == 0
    assert "Usage" in body.get("stdout", "")
