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
