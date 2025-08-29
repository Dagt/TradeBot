from dataclasses import dataclass

@dataclass
class Order:
    symbol: str
    side: str    # buy/sell
    type_: str   # market/limit
    qty: float
    price: float | None = None
    post_only: bool = False
    time_in_force: str | None = None
    iceberg_qty: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    reduce_only: bool = False
