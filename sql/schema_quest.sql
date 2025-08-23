-- QuestDB schema for crypto data

CREATE TABLE IF NOT EXISTS trades (
    ts TIMESTAMP,
    exchange SYMBOL,
    symbol SYMBOL,
    px DOUBLE,
    qty DOUBLE,
    side SYMBOL,
    trade_id STRING
) timestamp(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS orderbook (
    ts TIMESTAMP,
    exchange SYMBOL,
    symbol SYMBOL,
    bid_px STRING,
    bid_qty STRING,
    ask_px STRING,
    ask_qty STRING
) timestamp(ts) PARTITION BY DAY;

-- Best bid/ask snapshots
CREATE TABLE IF NOT EXISTS bba (
    ts TIMESTAMP,
    exchange SYMBOL,
    symbol SYMBOL,
    bid_px DOUBLE,
    bid_qty DOUBLE,
    ask_px DOUBLE,
    ask_qty DOUBLE
) timestamp(ts) PARTITION BY DAY;

-- Order book delta updates (price/qty lists stored as JSON strings)
CREATE TABLE IF NOT EXISTS book_delta (
    ts TIMESTAMP,
    exchange SYMBOL,
    symbol SYMBOL,
    bid_px STRING,
    bid_qty STRING,
    ask_px STRING,
    ask_qty STRING
) timestamp(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS bars (
    ts TIMESTAMP,
    timeframe SYMBOL,
    exchange SYMBOL,
    symbol SYMBOL,
    o DOUBLE,
    h DOUBLE,
    l DOUBLE,
    c DOUBLE,
    v DOUBLE
) timestamp(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS funding (
    ts TIMESTAMP,
    exchange SYMBOL,
    symbol SYMBOL,
    rate DOUBLE,
    interval_sec LONG
) timestamp(ts) PARTITION BY DAY;
