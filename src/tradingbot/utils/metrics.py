from prometheus_client import Counter, Histogram

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

# Order fills by symbol and side
FILL_COUNT = Counter(
    "order_fills",
    "Total order fills",
    ["symbol", "side"],
)

# Slippage observed in order execution (basis points)
SLIPPAGE = Histogram(
    "order_slippage_bps",
    "Distribution of order execution slippage in basis points",
    ["symbol", "side"],
)

# Risk management events triggered
RISK_EVENTS = Counter(
    "risk_events",
    "Total risk management events",
    ["event_type"],
)
