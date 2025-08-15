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
