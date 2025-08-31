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
  trade_id text,
  UNIQUE (ts, exchange, symbol)
);
SELECT create_hypertable('market.trades', by_range('ts'), if_not_exists => TRUE);

-- Orderbook snapshots
CREATE TABLE IF NOT EXISTS market.orderbook (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  bid_px numeric[] NOT NULL,
  bid_qty numeric[] NOT NULL,
  ask_px numeric[] NOT NULL,
  ask_qty numeric[] NOT NULL
);
SELECT create_hypertable('market.orderbook', by_range('ts'), if_not_exists => TRUE);

-- Best bid/ask snapshots (single level of the order book)
CREATE TABLE IF NOT EXISTS market.bba (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  bid_px numeric,
  bid_qty numeric,
  ask_px numeric,
  ask_qty numeric
);
SELECT create_hypertable('market.bba', by_range('ts'), if_not_exists => TRUE);

-- Order book deltas (price and quantity arrays representing incremental updates)
CREATE TABLE IF NOT EXISTS market.book_delta (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  bid_px numeric[] NOT NULL,
  bid_qty numeric[] NOT NULL,
  ask_px numeric[] NOT NULL,
  ask_qty numeric[] NOT NULL
);
SELECT create_hypertable('market.book_delta', by_range('ts'), if_not_exists => TRUE);

-- Bars
CREATE TABLE IF NOT EXISTS market.bars (
  ts timestamptz NOT NULL,
  timeframe text NOT NULL,   -- e.g. 1s,3m,5m
  exchange text NOT NULL,
  symbol text NOT NULL,
  o numeric, h numeric, l numeric, c numeric, v numeric,
  UNIQUE (ts, timeframe, exchange, symbol)
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

-- Open Interest (perps/spot)
CREATE TABLE IF NOT EXISTS market.open_interest (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  oi numeric NOT NULL
);
SELECT create_hypertable('market.open_interest', by_range('ts'), if_not_exists => TRUE);

-- Basis (perps vs spot)
CREATE TABLE IF NOT EXISTS market.basis (
  ts timestamptz NOT NULL,
  exchange text NOT NULL,
  symbol text NOT NULL,
  basis numeric NOT NULL
);
SELECT create_hypertable('market.basis', by_range('ts'), if_not_exists => TRUE);

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
CREATE INDEX IF NOT EXISTS idx_orders_ts ON market.orders(ts);

-- Triangular Arbitrage signals (persistencia de oportunidades)
CREATE TABLE IF NOT EXISTS market.tri_signals (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  exchange text NOT NULL,  -- e.g. 'binance'
  base text NOT NULL,
  mid text NOT NULL,
  quote text NOT NULL,
  direction text NOT NULL,    -- 'b->m' o 'm->b'
  edge numeric NOT NULL,      -- edge neto (decimal, ej 0.001 = 0.1%)
  notional_quote numeric NOT NULL,
  taker_fee_bps numeric NOT NULL,
  buffer_bps numeric NOT NULL,
  bq numeric NOT NULL,        -- precio BASE/QUOTE
  mq numeric NOT NULL,        -- precio MID/QUOTE
  mb numeric NOT NULL         -- precio MID/BASE
);
CREATE INDEX IF NOT EXISTS idx_tri_signals_ts ON market.tri_signals(ts);

-- Cross exchange arbitrage opportunities
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
CREATE INDEX IF NOT EXISTS idx_cross_signals_ts ON market.cross_signals(ts);

-- Portfolio exposure snapshots (por símbolo)
CREATE TABLE IF NOT EXISTS market.portfolio_snapshots (
  ts timestamptz NOT NULL DEFAULT now(),
  venue text NOT NULL,          -- ej: binance_spot_testnet / binance_futures_um_testnet
  symbol text NOT NULL,         -- BTC/USDT
  position numeric NOT NULL,    -- qty base (spot: >=0; futures: puede ser +/-)
  price numeric NOT NULL,       -- último precio usado para valorar
  notional_usd numeric NOT NULL -- |position| * price (aprox en USD/USDT)
);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_ts ON market.portfolio_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_venue ON market.portfolio_snapshots(venue);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_symbol ON market.portfolio_snapshots(symbol);

-- Risk events (violaciones/cierres auto)
CREATE TABLE IF NOT EXISTS market.risk_events (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  venue text NOT NULL,
  symbol text NOT NULL,
  kind text NOT NULL,        -- 'VIOLATION' | 'AUTO_CLOSE' | 'INFO'
  message text NOT NULL,
  details jsonb
);
CREATE INDEX IF NOT EXISTS idx_risk_events_ts ON market.risk_events(ts);
CREATE INDEX IF NOT EXISTS idx_risk_events_venue ON market.risk_events(venue);
CREATE INDEX IF NOT EXISTS idx_risk_events_symbol ON market.risk_events(symbol);

-- Posiciones por símbolo (acumuladas; avg_cost en base del símbolo)
CREATE TABLE IF NOT EXISTS market.positions (
  venue text NOT NULL,
  symbol text NOT NULL,
  qty numeric NOT NULL,            -- qty base (spot >=0; futures +/-)
  avg_price numeric NOT NULL,      -- costo promedio (USDT por unidad base)
  realized_pnl numeric NOT NULL,   -- PnL realizado acumulado (USDT)
  fees_paid numeric NOT NULL,      -- fees acumuladas (USDT)
  PRIMARY KEY (venue, symbol)
);

-- Snapshots de PnL (para dashboard/series)
CREATE TABLE IF NOT EXISTS market.pnl_snapshots (
  ts timestamptz NOT NULL DEFAULT now(),
  venue text NOT NULL,
  symbol text NOT NULL,
  qty numeric NOT NULL,
  price numeric NOT NULL,          -- último mark
  avg_price numeric NOT NULL,
  upnl numeric NOT NULL,           -- (mark - avg)*qty   (futures con signo de qty)
  rpnl numeric NOT NULL,           -- realized acumulado en positions
  fees numeric NOT NULL            -- fees acumuladas
);
CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_ts ON market.pnl_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_venue ON market.pnl_snapshots(venue);
CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_symbol ON market.pnl_snapshots(symbol);

-- Fills ejecutados (reales o simulados)
CREATE TABLE IF NOT EXISTS market.fills (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  venue text NOT NULL,             -- ej: binance_spot_testnet / binance_futures_um_testnet
  strategy text NOT NULL,          -- ej: breakout_atr_spot
  symbol text NOT NULL,            -- BTC/USDT
  side text NOT NULL,              -- buy/sell
  type text NOT NULL,              -- market/limit
  qty numeric NOT NULL,            -- cantidad base
  price numeric,                   -- precio de ejecución (si lo conocemos)
  notional numeric,                -- qty*price (si aplica)
  fee_usdt numeric,                -- costo estimado/real en USDT (si aplica)
  reduce_only boolean DEFAULT false,
  ext_order_id text,               -- id de orden del exchange (si aplica)
  raw jsonb                        -- payload crudo (para forense)
);
CREATE INDEX IF NOT EXISTS idx_fills_ts ON market.fills(ts);
CREATE INDEX IF NOT EXISTS idx_fills_venue ON market.fills(venue);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON market.fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_strategy ON market.fills(strategy);

