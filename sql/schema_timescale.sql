-- Timescale schema for crypto ticks/orderbook/bars/funding

CREATE SCHEMA IF NOT EXISTS market;

-- Trades
CREATE TABLE IF NOT EXISTS market.trades (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  px numeric NOT NULL,
  qty numeric NOT NULL,
  side text,        -- buy/sell if available
  trade_id text
);
SELECT create_hypertable('market.trades', by_range('ts'), if_not_exists => TRUE);

-- Bars
CREATE TABLE IF NOT EXISTS market.bars (
  ts timestamptz NOT NULL,
  timeframe text NOT NULL,   -- e.g. 1s,1m,5m
  exchange text NOT NULL,
  symbol text NOT NULL,
  o numeric, h numeric, l numeric, c numeric, v numeric
);
SELECT create_hypertable('market.bars', by_range('ts'), if_not_exists => TRUE);

-- Funding (perps)
CREATE TABLE IF NOT EXISTS market.funding (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  rate numeric NOT NULL,
  interval_sec int NOT NULL
);
SELECT create_hypertable('market.funding', by_range('ts'), if_not_exists => TRUE);

-- Orders/Executions (paper & live tracking)
CREATE TABLE IF NOT EXISTS market.orders (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  strategy text NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  side text NOT NULL, -- buy/sell
  type text NOT NULL, -- limit/market/...
  qty numeric NOT NULL,
  px numeric,
  status text NOT NULL, -- new/filled/partial/canceled/rejected
  ext_order_id text,
  notes jsonb
);
