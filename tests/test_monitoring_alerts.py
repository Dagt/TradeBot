import yaml

from tradingbot.utils.metrics import ORDER_BOOK_MIN_DEPTH, WS_RECONNECTS


def test_alerts_and_metrics_definitions():
    with open("monitoring/alerts.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rules = [rule for group in config.get("groups", []) for rule in group.get("rules", [])]
    rule_names = {rule["alert"]: rule for rule in rules}

    assert "LowOrderBookDepth" in rule_names
    assert "WebsocketReconnections" in rule_names
    assert "ExcessiveSlippage" in rule_names
    assert "RiskEvents" in rule_names
    assert "KillSwitchActive" in rule_names
    assert "WebsocketDisconnects" in rule_names

    assert "order_book_min_depth" in rule_names["LowOrderBookDepth"]["expr"]
    assert "ws_reconnections_total" in rule_names["WebsocketReconnections"]["expr"]
    assert "order_slippage_bps" in rule_names["ExcessiveSlippage"]["expr"]
    assert "risk_events_total" in rule_names["RiskEvents"]["expr"]
    assert "kill_switch_active" in rule_names["KillSwitchActive"]["expr"]
    assert "ws_failures_total" in rule_names["WebsocketDisconnects"]["expr"]

    ORDER_BOOK_MIN_DEPTH.clear()
    WS_RECONNECTS.clear()

    ORDER_BOOK_MIN_DEPTH.labels(symbol="BTCUSD", side="bid").set(5)
    depth_samples = list(ORDER_BOOK_MIN_DEPTH.collect())[0].samples
    depth_sample = [
        s for s in depth_samples
        if s.name == "order_book_min_depth"
        and s.labels["symbol"] == "BTCUSD"
        and s.labels["side"] == "bid"
    ][0]
    assert depth_sample.value == 5

    WS_RECONNECTS.labels(adapter="test").inc()
    ws_samples = list(WS_RECONNECTS.collect())[0].samples
    ws_sample = [
        s for s in ws_samples
        if s.name == "ws_reconnections_total" and s.labels["adapter"] == "test"
    ][0]
    assert ws_sample.value == 1.0
