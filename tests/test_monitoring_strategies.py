from monitoring import strategies


def test_register_and_params():
    strategies.register_strategy("alpha")
    assert "alpha" in strategies.available_strategies()["strategies"]

    strategies.update_strategy_params("alpha", {"n": 42})
    params = strategies.get_strategy_params("alpha")
    assert params["params"]["n"] == 42

    strategies.set_strategy_status("alpha", "running")
    status = strategies.strategies_status()["strategies"]["alpha"]["status"]
    assert status == "running"
