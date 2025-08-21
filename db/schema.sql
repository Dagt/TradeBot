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

CREATE UNIQUE INDEX IF NOT EXISTS trades_uq_idx
  ON market.trades (ts, exchange, symbol, trade_id);

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

CREATE UNIQUE INDEX IF NOT EXISTS bars_uq_idx
  ON market.bars (ts, timeframe, exchange, symbol);

CREATE TABLE IF NOT EXISTS market.funding (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  rate numeric NOT NULL,
  interval_sec int NOT NULL
);

CREATE TABLE IF NOT EXISTS market.open_interest (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  oi numeric NOT NULL
);

CREATE TABLE IF NOT EXISTS market.basis (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  basis numeric NOT NULL
);

CREATE TABLE IF NOT EXISTS market.orders (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  strategy text NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  side text NOT NULL,
  type text NOT NULL,
  qty numeric NOT NULL,
  px numeric,
  status text NOT NULL,
  ext_order_id text,
  notes jsonb
);

CREATE TABLE IF NOT EXISTS market.tri_signals (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  exchange text NOT NULL,
  base text NOT NULL,
  mid text NOT NULL,
  quote text NOT NULL,
  direction text NOT NULL,
  edge numeric NOT NULL,
  notional_quote numeric NOT NULL,
  taker_fee_bps numeric NOT NULL,
  buffer_bps numeric NOT NULL,
  bq numeric NOT NULL,
  mq numeric NOT NULL,
  mb numeric NOT NULL
);

CREATE TABLE IF NOT EXISTS market.cross_signals (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  symbol text NOT NULL,
  spot_exchange text NOT NULL,
  perp_exchange text NOT NULL,
  spot_px numeric NOT NULL,
  perp_px numeric NOT NULL,
  edge numeric NOT NULL
);
