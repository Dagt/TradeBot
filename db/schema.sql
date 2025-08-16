CREATE SCHEMA IF NOT EXISTS market;

CREATE TABLE IF NOT EXISTS market.trades (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  px numeric NOT NULL,
  qty numeric NOT NULL,
  side text,
  trade_id text
);

CREATE TABLE IF NOT EXISTS market.orderbook (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  bid_px numeric[] NOT NULL,
  bid_qty numeric[] NOT NULL,
  ask_px numeric[] NOT NULL,
  ask_qty numeric[] NOT NULL
);

CREATE TABLE IF NOT EXISTS market.bars (
  ts timestamptz NOT NULL,
  timeframe text NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  o numeric,
  h numeric,
  l numeric,
  c numeric,
  v numeric
);

CREATE TABLE IF NOT EXISTS market.funding (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  rate numeric NOT NULL,
  interval_sec int NOT NULL
);
