from fastapi.testclient import TestClient

from tradingbot.apps.api.main import app


def test_smoke():
    assert 2 + 2 == 4


def test_metrics_endpoint_exposed():
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "python_info" in resp.text
