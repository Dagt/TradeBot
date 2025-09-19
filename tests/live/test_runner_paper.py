import logging

from tradingbot.live import runner_paper


def test_exposure_metrics_logged_once_for_same_values(caplog):
    """Repeated ack/paper events with same exposure should log only once."""

    runner_paper._clear_exposure_log_registry()
    symbol = "BTC/USDT"
    side = "buy"
    exposure = 1.234
    locked = 5.678

    caplog.set_level(logging.INFO)
    caplog.clear()

    # Simulate on_order_ack and a repeated handle_paper_event with same state.
    first_logged = runner_paper._log_exposure_if_changed(symbol, side, exposure, locked)
    second_logged = runner_paper._log_exposure_if_changed(symbol, side, exposure, locked)

    assert first_logged is True
    assert second_logged is False

    exposure_logs = [
        record.message
        for record in caplog.records
        if record.message.startswith('METRICS {"exposure"')
    ]
    assert len(exposure_logs) == 1

    # Once the order is completed/cancelled the cache is cleared and logging resumes.
    runner_paper._reset_exposure_log(symbol, side)
    caplog.clear()

    again_logged = runner_paper._log_exposure_if_changed(symbol, side, exposure, locked)
    assert again_logged is True

    exposure_logs_after_reset = [
        record.message
        for record in caplog.records
        if record.message.startswith('METRICS {"exposure"')
    ]
    assert len(exposure_logs_after_reset) == 1

    runner_paper._clear_exposure_log_registry()
