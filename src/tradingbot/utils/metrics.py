import asyncio

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

# Current profit and loss in USD
TRADING_PNL = Gauge(
    "trading_pnl",
    "Current trading profit and loss in USD",
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

    async def _runner() -> None:
        while True:
            try:
                TRADING_PNL.set(getattr(broker.state, "realized_pnl", 0.0))

                positions = getattr(broker.state, "pos", {}) or {}
                for sym, pos in positions.items():
                    qty = getattr(pos, "qty", 0.0)
                    OPEN_POSITIONS.labels(symbol=sym).set(qty)

                existing = {
                    sample.labels["symbol"]
                    for metric in OPEN_POSITIONS.collect()
                    for sample in metric.samples
                    if sample.name == "open_position"
                }
                for sym in existing - positions.keys():
                    OPEN_POSITIONS.labels(symbol=sym).set(0.0)
            except Exception:  # pragma: no cover - best effort
                pass
            await asyncio.sleep(interval)

    loop.create_task(_runner())
