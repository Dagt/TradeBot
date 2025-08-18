from __future__ import annotations
import os
import logging

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback if package missing
    def load_dotenv() -> None:  # type: ignore
        """Fallback no-op if python-dotenv is not installed."""
        return

from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
from ..adapters.binance_spot import BinanceSpotAdapter
from ..execution.paper import PaperAdapter

log = logging.getLogger(__name__)


async def run_real(
    symbol: str = "BTC/USDT",
    trade_qty: float = 0.001,
    *,
    dry_run: bool = False,
    i_know_what_im_doing: bool = False,
) -> None:
    """Run a very small live trading loop against real endpoints.

    Parameters
    ----------
    symbol:
        Trading pair to operate.
    trade_qty:
        Quantity to trade when a tick is received.
    dry_run:
        If ``True`` the :class:`PaperAdapter` is used instead of a real
        exchange adapter and no real orders are sent.
    i_know_what_im_doing:
        Mandatory confirmation flag to avoid accidental live runs.
    """

    if not i_know_what_im_doing:
        raise RuntimeError("Refusing to trade live without --i-know-what-im-doing")

    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    ws = BinanceSpotWSAdapter()
    broker = PaperAdapter() if dry_run else BinanceSpotAdapter(api_key=api_key, api_secret=api_secret)

    async for _ in ws.stream_trades(symbol):
        await broker.place_order(symbol, "buy", "market", trade_qty)
        break
