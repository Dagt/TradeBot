def test_backtest_param_config_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGBOT_SKIP_NTP_CHECK", "1")
    from tradingbot.cli.commands import backtesting as cli_backtesting
    from tradingbot.backtest import event_engine as ev_module
    from tradingbot import strategies as strat_module

    datafile = tmp_path / "data.csv"
    datafile.write_text("ts,o,h,l,c,v\n0,1,1,1,1,1\n")
    cfgfile = tmp_path / "cfg.yaml"
    cfgfile.write_text("{}")

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            self.strategies = {}
        def run(self, fills_csv=None):
            return {}

    monkeypatch.setattr(ev_module, "EventDrivenBacktestEngine", DummyEngine)
    monkeypatch.setattr(cli_backtesting, "generate_report", lambda result: "")

    captured = {}

    class DummyStrategy:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def on_bar(self, bar):
            pass

    monkeypatch.setitem(strat_module.STRATEGIES, "dummy", DummyStrategy)

    cli_backtesting.backtest(
        data=str(datafile),
        symbol="BTC/USDT",
        strategy="dummy",
        config=None,
        param=[f"config_path={cfgfile}"]
    )

    assert captured["config_path"] == str(cfgfile)

