from tradingbot.live.paper_orders import PaperOrderManager


def _collect_events():
    events: list[dict] = []

    def emit(payload: dict) -> None:
        events.append(payload)

    return events, emit


def test_order_cancel_resets_locked():
    events, emit = _collect_events()
    exposures = {"BTC/USDT": 0.0}
    manager = PaperOrderManager(emit=emit, exposure_fn=lambda sym: exposures.get(sym, 0.0))

    manager.on_order(
        symbol="BTC/USDT",
        order_id="1",
        side="buy",
        price=100.0,
        orig_qty=1.0,
        remaining_qty=1.0,
    )
    assert manager.locked_total == 100.0

    exposures["BTC/USDT"] = 0.0
    manager.on_cancel(symbol="BTC/USDT", order_id="1", reason="expired")

    assert manager.locked_total == 0.0
    assert [ev.get("event") for ev in events if "event" in ev] == ["order", "cancel"]
    exposures_events = [ev for ev in events if "locked" in ev]
    assert exposures_events[-2]["locked"] == 100.0
    assert exposures_events[-1]["locked"] == 0.0


def test_partial_fills_reduce_locked():
    events, emit = _collect_events()
    exposures = {"ETH/USDT": 0.0}
    manager = PaperOrderManager(emit=emit, exposure_fn=lambda sym: exposures.get(sym, 0.0))

    manager.on_order(
        symbol="ETH/USDT",
        order_id="order-1",
        side="buy",
        price=10.0,
        orig_qty=10.0,
        remaining_qty=10.0,
    )
    assert manager.locked_total == 100.0

    exposures["ETH/USDT"] = 4.0
    manager.on_fill(
        symbol="ETH/USDT",
        order_id="order-1",
        side="buy",
        fill_qty=4.0,
        price=10.0,
        fee=0.01,
        pending_qty=6.0,
        maker=True,
        slippage_bps=0.0,
    )
    assert manager.locked_total == 60.0

    exposures["ETH/USDT"] = 10.0
    manager.on_fill(
        symbol="ETH/USDT",
        order_id="order-1",
        side="buy",
        fill_qty=6.0,
        price=10.0,
        fee=0.02,
        pending_qty=0.0,
        maker=False,
        slippage_bps=0.0,
    )
    assert manager.locked_total == 0.0

    events_by_type = [ev.get("event") for ev in events if "event" in ev]
    assert events_by_type.count("order") == 1
    assert events_by_type.count("fill") == 2
    exposures = [ev["locked"] for ev in events if "locked" in ev]
    assert exposures[:3] == [100.0, 60.0, 0.0]


def test_delayed_fill_clears_locked():
    events, emit = _collect_events()
    exposures = {"ADA/USDT": 0.0}
    manager = PaperOrderManager(emit=emit, exposure_fn=lambda sym: exposures.get(sym, 0.0))

    manager.on_order(
        symbol="ADA/USDT",
        order_id="7",
        side="sell",
        price=2.0,
        orig_qty=5.0,
        remaining_qty=5.0,
    )
    assert manager.locked_total == 10.0

    exposures["ADA/USDT"] = -5.0
    manager.on_fill(
        symbol="ADA/USDT",
        order_id="7",
        side="sell",
        fill_qty=5.0,
        price=2.0,
        fee=0.0,
        pending_qty=0.0,
        maker=True,
        slippage_bps=0.0,
    )
    assert manager.locked_total == 0.0

    events_by_type = [ev.get("event") for ev in events if "event" in ev]
    assert events_by_type.count("order") == 1
    assert events_by_type.count("fill") == 1
    exposures = [ev["locked"] for ev in events if "locked" in ev]
    assert exposures[-2:] == [10.0, 0.0]


def test_emit_skip_records_event():
    events, emit = _collect_events()
    manager = PaperOrderManager(emit=emit, exposure_fn=lambda _sym: 0.0)

    manager.emit_skip("min_notional")

    assert events == [{"event": "skip", "reason": "min_notional"}]
    assert manager.open_orders == {}


def test_pending_none_updates_remaining():
    events, emit = _collect_events()
    manager = PaperOrderManager(emit=emit, exposure_fn=lambda _sym: 0.0)

    manager.on_order(
        symbol="SOL/USDT",
        order_id="abc",
        side="buy",
        price=20.0,
        orig_qty=5.0,
        remaining_qty=5.0,
    )
    manager.on_fill(
        symbol="SOL/USDT",
        order_id="abc",
        side="buy",
        fill_qty=2.0,
        price=20.0,
        fee=0.0,
        pending_qty=None,
        maker=False,
    )

    assert manager.locked_total == 60.0
    fill_events = [ev for ev in events if ev.get("event") == "fill"]
    assert fill_events[0]["qty"] == 2.0


def test_ensure_order_event_emits_once():
    events, emit = _collect_events()
    manager = PaperOrderManager(emit=emit, exposure_fn=lambda _sym: 0.0)

    manager.ensure_order_event(
        symbol="XRP/USDT",
        order_id="1",
        side="buy",
        price=0.5,
        orig_qty=100.0,
        remaining_qty=80.0,
    )
    manager.ensure_order_event(
        symbol="XRP/USDT",
        order_id="1",
        side="buy",
        price=0.5,
        orig_qty=100.0,
        remaining_qty=60.0,
    )

    events_by_type = [ev.get("event") for ev in events if "event" in ev]
    assert events_by_type.count("order") == 1
