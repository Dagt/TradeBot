import math


def limit_price_from_close(side: str, close: float, tick_size: float) -> float:
    """Return a limit price derived from the closed bar.

    In backtests the default limit price is simply the bar close.  Live runners
    should behave identically but must also respect the instrument ``tick_size``.
    ``tick_size`` of ``0`` leaves the price untouched which mirrors the
    backtest behaviour.

    Parameters
    ----------
    side:
        Order side (``"buy"`` or ``"sell"``) used to determine rounding
        direction when adjusting to ``tick_size``.
    close:
        Close price of the most recently completed bar.
    tick_size:
        Minimum price increment for the instrument.
    """

    price = close
    if tick_size > 0:
        if side == "buy":
            price = math.floor(close / tick_size) * tick_size
        else:
            price = math.ceil(close / tick_size) * tick_size
    return price
