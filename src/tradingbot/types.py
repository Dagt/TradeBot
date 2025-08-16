from dataclasses import dataclass
from datetime import datetime

@dataclass
class Tick:
    ts: datetime
    exchange: str
    symbol: str
    price: float
    qty: float
    side: str | None = None  # buy/sell

@dataclass
class Bar:
    ts: datetime
    timeframe: str
    exchange: str
    symbol: str
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class OrderBook:
    ts: datetime
    exchange: str
    symbol: str
    bid_px: list[float]
    bid_qty: list[float]
    ask_px: list[float]
    ask_qty: list[float]
