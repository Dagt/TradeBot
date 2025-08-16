import importlib

import pytest


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        """
EXCH_API_KEY_1=first
EXCH_API_SECRET_1=sec1
EXCH_API_KEY_2=second
EXCH_API_SECRET_2=sec2
""".strip()
    )
    monkeypatch.chdir(tmp_path)
    mod = importlib.import_module("tradingbot.secrets.manager")
    importlib.reload(mod)
    return mod.SecretsManager(rate_limit=1, per_seconds=60)


def test_rotation_and_rate_limit(manager):
    k1 = manager.get_api_credentials("exch")
    assert k1 == ("first", "sec1")
    # second call should rotate due to rate limit 1
    k2 = manager.get_api_credentials("exch")
    assert k2 == ("second", "sec2")


def test_logs_do_not_leak_secrets(manager, caplog):
    caplog.set_level("INFO")
    manager.get_api_credentials("exch")
    log = caplog.text
    assert "first" not in log
    assert "sec1" not in log
    assert "second" not in log
    assert "sec2" not in log


def test_validate_permissions_warns(manager, caplog):
    class Dummy:
        id = "dummy"
        has = {"fetchPermissions": True}

        def fetch_permissions(self):
            return {"read": True, "trade": True, "withdraw": True}

    caplog.set_level("WARNING")
    manager.validate_permissions(Dummy())
    assert any("withdraw" in r.message for r in caplog.records)
