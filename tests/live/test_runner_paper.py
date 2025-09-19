import logging

from tradingbot.live import runner_paper


def test_exposure_metrics_dedup_for_repeated_ack_and_paper_events(caplog):
    """Simulate ack/paper events and ensure repeated states log exposure once."""

    runner_paper._clear_exposure_log_registry()
    symbol = "BTC/USDT"
    side = "buy"
    exposure = 1.234
    locked = 5.678

    caplog.set_level(logging.INFO)
    caplog.clear()

    # First event represents an order acknowledgement.
    ack_logged = runner_paper._log_exposure_if_changed(symbol, side, exposure, locked)

    # Subsequent paper events with the same snapshot should not re-log the line.
    first_paper_logged = runner_paper._log_exposure_if_changed(
        symbol, side, exposure, locked
    )
    second_paper_logged = runner_paper._log_exposure_if_changed(
        symbol, side, exposure, locked
    )

    assert ack_logged is True
    assert first_paper_logged is False
    assert second_paper_logged is False

    exposure_logs = [
        record.message
        for record in caplog.records
        if record.message.startswith('METRICS {"exposure"')
    ]
    assert len(exposure_logs) == 1

    # Once the order is completed/cancelled the cache is cleared and logging resumes.
    runner_paper._reset_exposure_log(symbol, side)
    caplog.clear()

    ack_after_reset = runner_paper._log_exposure_if_changed(
        symbol, side, exposure, locked
    )
    paper_after_reset = runner_paper._log_exposure_if_changed(
        symbol, side, exposure, locked
    )

    assert ack_after_reset is True
    assert paper_after_reset is False

    exposure_logs_after_reset = [
        record.message
        for record in caplog.records
        if record.message.startswith('METRICS {"exposure"')
    ]
    assert len(exposure_logs_after_reset) == 1

    runner_paper._clear_exposure_log_registry()
