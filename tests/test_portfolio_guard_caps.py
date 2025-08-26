from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig


def test_would_exceed_caps_optional_params():
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=0.5, venue="X"))
    guard.refresh_usd_caps(100.0)
    guard.mark_price("BTC", 10.0)
    guard.set_position("X", "BTC", 2.0)

    exceed, _, _ = guard.would_exceed_caps("BTC", "buy")
    assert exceed is False

    exceed, _, _ = guard.would_exceed_caps("BTC", "buy", add_qty=4.0, price=10.0)
    assert exceed is True

