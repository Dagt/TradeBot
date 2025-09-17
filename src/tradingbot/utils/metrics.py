import asyncio
from collections.abc import Mapping

from prometheus_client import Counter, Histogram, Gauge

# Latency of HTTP requests by method and endpoint
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["method", "endpoint"],
)

# Total HTTP requests processed
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)

# Websocket connection failures by adapter
WS_FAILURES = Counter(
    "ws_failures_total",
    "Total websocket connection failures",
    ["adapter"],
)

# Websocket reconnections by adapter
WS_RECONNECTS = Counter(
    "ws_reconnections_total",
    "Total websocket reconnections",
    ["adapter"],
)

# Order fills by symbol and side
FILL_COUNT = Counter(
    "order_fills",
    "Total order fills",
    ["symbol", "side"],
)

# Order flow counters
ORDERS = Counter(
    "orders_total",
    "Total orders submitted to the broker",
)

CANCELS = Counter(
    "order_cancels_total",
    "Total order cancellations or expiries",
)

SKIPS = Counter(
    "order_skips_total",
    "Total orders skipped before submission",
)

# Slippage observed in order execution (basis points)
SLIPPAGE = Histogram(
    "order_slippage_bps",
    "Distribution of order execution slippage in basis points",
    ["symbol", "side"],
)

# Position in queue at top of book as ratio (0-1)
QUEUE_POSITION = Histogram(
    "order_queue_position_ratio",
    "Fraction of existing liquidity ahead of the order at best price level",
    ["symbol", "side"],
)

# Minimum depth available at top of book
ORDER_BOOK_MIN_DEPTH = Gauge(
    "order_book_min_depth",
    "Minimum order book depth at best price level",
    ["symbol", "side"],
)

# Risk management events triggered
RISK_EVENTS = Counter(
    "risk_events",
    "Total risk management events",
    ["event_type"],
)

# Kill switch status (1 = halted)
KILL_SWITCH_ACTIVE = Gauge(
    "kill_switch_active",
    "Global kill switch active flag",
)

# Execution latency by venue
ORDER_LATENCY = Histogram(
    "order_execution_latency_seconds",
    "Latency of order execution in seconds",
    ["venue"],
)

# Maker/taker ratio per venue
MAKER_TAKER_RATIO = Gauge(
    "maker_taker_ratio",
    "Ratio of maker to taker orders",
    ["venue"],
)

# Venue selection metrics
ROUTER_SELECTED_VENUE = Counter(
    "router_selected_venue_total",
    "Total venue selections by router",
    ["venue", "path"],
)

ROUTER_STALE_BOOK = Counter(
    "router_stale_book_total",
    "Venues skipped due to stale order book data",
    ["venue"],
)

# --- Additional high level trading metrics ---

# Current profit and loss in USD (realized component)
TRADING_PNL = Gauge(
    "trading_pnl",
    "Current realized trading profit and loss in USD",
)

# Unrealized profit and loss in USD
TRADING_PNL_UNREALIZED = Gauge(
    "trading_pnl_unrealized",
    "Current unrealized trading profit and loss in USD",
)

# Total profit and loss in USD
TRADING_PNL_TOTAL = Gauge(
    "trading_pnl_total",
    "Current total trading profit and loss in USD",
)

# Open position size per symbol
OPEN_POSITIONS = Gauge(
    "open_position",
    "Current open position size",
    ["symbol"],
)

# Funding rate per symbol
FUNDING_RATE = Gauge(
    "funding_rate",
    "Latest funding rate",
    ["symbol"],
)

FUNDING_RATE_HIST = Histogram(
    "funding_rate_distribution",
    "Distribution of funding rate observations",
    ["symbol"],
)

# Basis per symbol
BASIS = Gauge(
    "basis",
    "Latest basis value",
    ["symbol"],
)

BASIS_HIST = Histogram(
    "basis_distribution",
    "Distribution of basis observations",
    ["symbol"],
)

# Open interest per symbol
OPEN_INTEREST = Gauge(
    "open_interest",
    "Current open interest",
    ["symbol"],
)

OPEN_INTEREST_HIST = Histogram(
    "open_interest_distribution",
    "Distribution of open interest observations",
    ["symbol"],
)

# Latency of market data processing
MARKET_LATENCY = Histogram(
    "market_latency_seconds",
    "Latency of market data processing in seconds",
)

# Completed bars accumulated by the bar aggregator
AGG_COMPLETED = Gauge(
    "aggregated_bars",
    "Number of completed bars accumulated by the bar aggregator",
)

# End-to-end order processing latency
E2E_LATENCY = Histogram(
    "e2e_latency_seconds",
    "End-to-end order processing latency in seconds",
)

# Latency from signal generation to order confirmation
SIGNAL_CONFIRM_LATENCY = Histogram(
    "signal_confirmation_latency_seconds",
    "Latency from signal generation to order confirmation in seconds",
)

# Order book persistence failures
ORDERBOOK_INSERT_FAILURES = Counter(
    "orderbook_insert_failures_total",
    "Total order book persistence failures",
)


def start_pnl_position_updater(broker, interval: float = 5.0) -> None:
    """Launch background task updating PnL and position gauges.

    Parameters
    ----------
    broker:
        Trading broker or adapter exposing ``state.realized_pnl`` and a
        mapping ``state.pos`` with position objects containing ``qty``.
    interval:
        Seconds between metric updates.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no running loop
        return

    def _extract_numeric(
        source: object,
        keys: tuple[str, ...],
        *,
        default: float | None = 0.0,
    ) -> float | None:
        """Best-effort retrieval of a numeric attribute from ``source``.

        The helper accepts both mappings and objects with attributes.  The
        first key found with a non-``None`` value is returned as ``float``.
        If no key matches, ``default`` is returned which may itself be
        ``None`` to signal missing data.
        """

        if isinstance(source, Mapping):
            for key in keys:
                if key in source and source[key] is not None:
                    try:
                        return float(source[key])
                    except (TypeError, ValueError):
                        continue
        for key in keys:
            if hasattr(source, key):
                value = getattr(source, key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return default

    async def _runner() -> None:
        while True:
            try:
                realized = float(getattr(broker.state, "realized_pnl", 0.0) or 0.0)
                TRADING_PNL.set(realized)

                positions = getattr(broker.state, "pos", {}) or {}
                last_prices = getattr(broker.state, "last_px", {}) or {}
                pending_fees_map: Mapping[str, float] = {}
                for attr in ("fees_pending", "pending_fees", "fees_due"):
                    candidate = getattr(broker.state, attr, None)
                    if isinstance(candidate, Mapping):
                        pending_fees_map = candidate  # type: ignore[assignment]
                        break

                unrealized_total = 0.0
                positions = getattr(broker.state, "pos", {}) or {}
                for sym, pos in positions.items():
                    qty = _extract_numeric(pos, ("qty", "position", "position_amt", "positionAmt", "size"), default=0.0) or 0.0
                    OPEN_POSITIONS.labels(symbol=sym).set(qty)
                    if abs(qty) <= 1e-12:
                        continue

                    avg_price = _extract_numeric(
                        pos,
                        (
                            "avg_price",
                            "avg_px",
                            "entry_price",
                            "avgEntryPrice",
                            "avg_entry",
                        ),
                        default=None,
                    )
                    if avg_price is None:
                        continue

                    last_px = None
                    if isinstance(last_prices, Mapping):
                        raw_last = last_prices.get(sym)
                        if raw_last is not None:
                            try:
                                last_px = float(raw_last)
                            except (TypeError, ValueError):
                                last_px = None
                    if last_px is None:
                        last_px = _extract_numeric(
                            pos,
                            (
                                "mark_price",
                                "last_price",
                                "price",
                                "close",
                            ),
                            default=None,
                        )
                    if last_px is None:
                        continue

                    pending_fee = _extract_numeric(
                        pos,
                        ("fees_pending", "pending_fees", "pending_fee"),
                        default=0.0,
                    ) or 0.0
                    if isinstance(pending_fees_map, Mapping):
                        extra_fee = pending_fees_map.get(sym)
                        if extra_fee is not None:
                            try:
                                pending_fee += float(extra_fee)
                            except (TypeError, ValueError):
                                pass

                    unrealized_total += qty * (last_px - avg_price) - pending_fee

                existing = {
                    sample.labels["symbol"]
                    for metric in OPEN_POSITIONS.collect()
                    for sample in metric.samples
                    if sample.name == "open_position"
                }
                for sym in existing - positions.keys():
                    OPEN_POSITIONS.labels(symbol=sym).set(0.0)

                TRADING_PNL_UNREALIZED.set(unrealized_total)
                TRADING_PNL_TOTAL.set(realized + unrealized_total)
            except Exception:  # pragma: no cover - best effort
                pass
            await asyncio.sleep(interval)

    loop.create_task(_runner())
